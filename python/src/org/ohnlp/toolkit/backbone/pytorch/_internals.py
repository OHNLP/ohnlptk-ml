from typing import List, Union

from ohnlp.toolkit.backbone.api import BackboneComponent, Schema


class PytorchDualTaskComponentWrapper(BackboneComponent):

    def init(self, configstr: Union[str, None]) -> None:
        pass

    def to_do_fn_config(self) -> str:
        pass

    def get_input_tag(self) -> str:
        pass

    def get_output_tags(self) -> List[str]:
        pass

    def calculate_output_schema(self, input_schema: dict[str, Schema]) -> dict[str, Schema]:
        pass