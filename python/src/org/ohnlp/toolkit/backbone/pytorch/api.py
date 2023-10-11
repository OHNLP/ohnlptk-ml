import asyncio
import pickle
import signal
import uuid
from abc import ABC, abstractmethod
from ast import Bytes
from asyncio import Future, Lock, Event
from typing import List, Any, Iterator, Union, Dict, Type, Iterable, Optional

import torch.nn
from ohnlp.toolkit.backbone.api import Row, Schema, ModuleDeclaration, UserDefinedPartitionMappingFunction, \
    OutputCollector, Transform, PartitionedRowCollectionTuple, Field, FieldType, TypeName, UserDefinedReductionFunction, \
    UDF_IN_TYPE, ManyToOneTransform, PartitionedCollection
from ohnlp.toolkit.backbone.backbone_module_launcher import find_free_port
from torch import Tensor
from torch.nn import Parameter
from torch.optim import Optimizer
from websockets.client import connect
from websockets.server import serve, ServerConnection

from python.src.org.ohnlp.toolkit.backbone.pytorch.loss_aggregators import LossAggregationStrategy


class PytorchModelModuleDefinition(ModuleDeclaration, ABC):
    @abstractmethod
    def load_model(self) -> torch.nn.Module:
        pass


class PytorchTrainWebsocketServer:
    r"""
    Hub component of federated model training, aggregates weights across various training instances.
    Model training is done using the fedAVG algorithm
    """
    # Server start/stop
    run_flag: bool = True
    stop: Future = asyncio.Future()

    # Training executor registry
    registry_lock: Lock = asyncio.Lock()
    registered_trainer_count: int = 0
    completed_trainers_curr_batch: int = 0

    # Model and Loss Tracking
    prior_batch_total_loss: Tensor = None
    epoch_total_loss: Tensor = None
    epoch_loss_accum_lock: Lock = asyncio.Lock()
    module: torch.nn.Module = None
    optimizer: Optimizer = None
    epoch: int = 0

    batch_complete_event: Event = asyncio.Event()

    def __init__(self, module: torch.nn.Module, optimizer: Optimizer, loss_aggregator: LossAggregationStrategy,
                 epoch: int, in_train_mode: bool, out_path):
        # Boot up a websocket server
        # - Find a port
        self.ws_port = find_free_port()
        # - Start a server
        asyncio.run(self.start_server())
        self.module = module
        self.module.train()
        self.optimizer = optimizer
        self.optimizer.zero_grad()
        self.loss_aggregator = loss_aggregator
        self.epoch = epoch
        self.persistence = out_path
        self.in_train_mode = in_train_mode

    async def handle(self, websocket: ServerConnection):
        async for message in websocket:
            parts: List[str] = message.split(' ')
            op: str = parts[0]
            client: str = parts[1]
            data: List[str] = []
            if len(parts) > 2:
                data = parts[2:]
            if op == 'register':
                async with self.registry_lock:
                    self.registered_trainer_count += 1
                    async with self.epoch_loss_accum_lock:
                        if self.prior_batch_total_loss is not None:
                            loss_raw = pickle.dumps(self.prior_batch_total_loss).hex()
                            websocket.send(f"init driver {loss_raw}")
                        else:
                            websocket.send("init driver")
            elif op == 'sync':
                await self.handle_model_sync(websocket, data)
            elif op == 'unregister':
                async with self.registry_lock:
                    self.registered_trainer_count -= 1

    async def start_server(self):
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGTERM, self.stop.set_result, None)
        async with serve(self.handle, "localhost", self.ws_port):
            await self.stop

    async def handle_model_sync(self, websocket: ServerConnection, data: List[str]):
        tensor_hex = data[1]
        client_loss_tensor: Tensor = pickle.loads(bytes.fromhex(tensor_hex))
        async with self.epoch_loss_accum_lock:
            self.loss_aggregator.add_executor_loss(client_loss_tensor)
        async with self.registry_lock:
            self.completed_trainers_curr_batch += 1
            if self.completed_trainers_curr_batch >= self.registered_trainer_count:
                self.completed_trainers_curr_batch = 0
                async with self.epoch_loss_accum_lock:
                    self.prior_batch_total_loss = self.loss_aggregator.agg_loss()
                    self.epoch_total_loss += self.prior_batch_total_loss
                    self.batch_complete_event.set()
                    self.return_loss_to_trainer(websocket)
                    self.loss_aggregator.reset_batch()
                    self.batch_complete_event = Event()
            else:
                await self.batch_complete_event.wait()
                self.return_loss_to_trainer(websocket)

    def return_loss_to_trainer(self, websocket: ServerConnection):
        loss_data: str = pickle.dumps(self.prior_batch_total_loss).hex()
        websocket.send(f"sync driver {loss_data}")

    def end_epoch(self) -> Tensor:
        if self.in_train_mode:
            self.epoch_total_loss.backward()
            torch.save(self.module.state_dict(), self.persistence + f'/model.{self.epoch}.pt')
        return self.epoch_total_loss

    def shutdown(self):
        # TODO write
        self.stop.set_result(None)
        pass


class PytorchTrainBatchDoFn(UserDefinedPartitionMappingFunction[Row, Bytes], ABC):
    r"""
    An abstract class to perform federated batch training on neural network models. Federation/loss aggregation is
    defined by the implementing class on a model-by-model basis, although some default implementations such as
    fedAVG are supplied.

    In essence, each "bundle" is treated as a mini-batch, and each full run through the dataset is
    treated as an epoch for model training purposes.

    Training is done in a data-parallel manner, where models are deployed in parallel and then data is sharded into
    mini-batches. Each parallel instance is then passed a single shard for training purposes. In backpropagation, the
    losses/associated gradients are aggregated depending on the implementation (fedAVG by default) and globally updated
    on all replicas at the conclusion of each mini-batch
    """
    driver_ws_conn: str = None
    ws_client_id: str = None

    model: torch.nn.Module = None
    output: List[Any] = []
    labels: List[Any] = []
    optimizer: Optimizer = None

    state_dict: Dict[str, Any]

    curr_epoch_batch_num: int = -1

    def __init__(self, component_def: PytorchModelModuleDefinition, in_train_mode: bool):
        super().__init__()
        self.component_def = component_def
        self.in_train_mode = in_train_mode

    def init_from_driver(self, json_config: Dict) -> None:
        # TODO: init driver_ws_conn
        if self.in_train_mode:
            self.driver_ws_conn = json_config['driver_ws']
        else:
            self.state_dict = pickle.loads(json_config['state_dict'])
        self.ws_client_id = str(uuid.uuid4())
        self.on_component_start()

    def on_component_start(self) -> None:
        # Equivalent to start of epoch, load model and register trainer with driver

        self.model = self.component_def.load_model()
        if self.in_train_mode:
            self.model.train()
            self.optimizer = self.get_optimizer(self.model.parameters())
            loss: Union[Tensor, None] = None
            with connect(self.driver_ws_conn) as websocket:
                websocket.send(f'register {self.ws_client_id}')
                resp = websocket.recv().split(' ')
                if len(resp) > 2:
                    loss: Tensor = pickle.loads(bytes.fromhex(resp[2]))
            if loss is not None:
                loss.backward()
        else:
            self.model.load_state_dict(self.state_dict)

    def on_bundle_start(self) -> None:
        # Equivalent to Start of next batch
        self.curr_epoch_batch_num += 1
        if self.in_train_mode:
            self.optimizer.zero_grad()

    def on_bundle_finish(self, out: OutputCollector) -> None:
        # Equivalent to finish batch/all batch elements processed
        loss: Tensor = self.calculate_loss(self.output, self.labels)
        # Output batch loss
        out.output(pickle.dumps(loss))
        # Broadcast loss to driver for summing if in train mode
        loss_serialized: str = pickle.dumps(loss).hex()
        if self.in_train_mode:
            with connect(self.driver_ws_conn) as websocket:
                websocket.send(f'sync {self.ws_client_id} {self.curr_epoch_batch_num} {loss_serialized}')
                recv = bytes.fromhex(websocket.recv().split(' ')[2])
                # TODO handle heartbeat
                # Get agg loss and backprop locally
                if self.in_train_mode:
                    loss = pickle.loads(recv)
                    loss.backward()
                    self.optimizer.step()

    def on_teardown(self) -> None:
        # Equivalent to end of epoch, unregister training instance from driver
        if self.in_train_mode:
            with connect(self.driver_ws_conn) as websocket:
                websocket.send(f'unregister {self.ws_client_id}')

    def apply(self, out: OutputCollector, input_row: Row) -> None:
        self.output.append(self.label_from_output(self.model(self.to_model_inputs(input_row))))
        self.labels.append(self.get_label(input_row))

    @abstractmethod
    def get_optimizer(self, parameters: Iterator[Parameter]):
        pass

    @abstractmethod
    def to_model_inputs(self, input_row: Row):
        pass

    @abstractmethod
    def get_label(self, input_row: Row):
        pass

    @abstractmethod
    def label_from_output(self, output):
        pass

    @abstractmethod
    def calculate_loss(self, output_labels: List[Any], data_labels: List[Any]) -> Tensor:
        pass


class LossAggregationReductionFunction(UserDefinedReductionFunction[bytes]):

    def init_from_driver(self, json_config: Optional[Dict]) -> None:
        pass

    def reduce(self, elements: Iterable[bytes]) -> bytes:
        pass


class PytorchTrainComponent(ManyToOneTransform):
    epoch_limit: int
    train_server: PytorchTrainWebsocketServer

    def __init__(self, component_def: PytorchModelModuleDefinition, epoch: int):
        super().__init__()
        self.component_def = component_def
        self.epoch = epoch

    def init(self) -> None:
        # Start a local websocket service for model training
        self.train_server = PytorchTrainWebsocketServer()
        pass

    def get_input_tags(self) -> List[str]:
        return ["Train", "Validation", "Test"]

    def get_output_tag(self) -> str:
        return "Model"

    def reduce(self, input_val: PartitionedRowCollectionTuple) -> PartitionedCollection[Row]:
        epoch: int = 0
        curr_ret = input_val
        train = input_val.get('Train')
        validation = input_val.get('Validation')
        test = input_val.get('Test')
        config = self.get_batch_training_config()
        config['driver_ws'] = self.train_server.ws_port  # TODO get actual connection info instead of just port
        loss_reduce_config = {}  # TODO
        while epoch < self.epoch_limit:
            train_losses = train.apply(f"Train Model (Epoch {epoch}/{self.epoch_limit}",
                                       self.get_batch_training_function_type()(self.component_def, True),
                                       # TODO pass arguments illegal, use config instead
                                       config)  # TODO add websocket and weights
            # Execute a reduction on train_losses to ensure this actually completes before calculating validation losses
            agg_train_loss = train_losses.reduce_global("Calculate accumulated loss",
                                                           LossAggregationReductionFunction(),
                                                           loss_reduce_config).first()  # TODO implement first
            # Reduce train_losses here to get
            val_losses = validation.apply(f"Validate Model (Epoch {epoch}/{self.epoch_limit}",
                                          self.get_batch_training_function_type()(self.component_def, False),
                                          # TODO pass arguments illegal, use config instead
                                          config)  # TODO need to add pickled model state_dict to config here
            # Execute another reduction on val losses
            agg_val_loss = val_losses.reduce_global("Calculate accumulated loss",
                                                       LossAggregationReductionFunction(),
                                                       loss_reduce_config).first()  # TODO implement first
            epoch += 1
            if self.check_early_stop(agg_val_loss):  # TODO
                break
        test_loss = test.apply(f"Evaluate Model",
                               self.get_batch_training_function_type()(self.component_def, False),
                               # TODO pass arguments illegal, use config instead
                               config)  # TODO need to add pickled model state_dict to config here
        state_dict = input_val
        return curr_ret

    def get_required_columns(self, input_tag: str) -> Union[Schema, None]:
        return None

    def calculate_output_schema(self, input_schemas: Dict[str, Schema]) -> Dict[str, Schema]:
        return input_schemas

    def teardown(self):
        self.train_server.shutdown()

    def to_java(self):
        return self._java_obj

    @abstractmethod
    def get_batch_training_function_type(self) -> Type[PytorchTrainBatchDoFn]:
        pass

    @abstractmethod
    def get_batch_training_config(self) -> Dict:
        pass
