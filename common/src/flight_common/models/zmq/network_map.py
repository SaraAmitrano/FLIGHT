from pydantic import BaseModel, RootModel


class PortMapping(RootModel[dict[str, int]]):
    def __getitem__(self, item: str) -> int:
        return self.root[item]

    def get(self, item: str) -> int | None:
        return self.root.get(item)


class NetworkData(BaseModel):
    api: PortMapping
    orchestrator: PortMapping


class RecvSendData(BaseModel):
    SEND: dict[str, str]
    RECV: dict[str, str]


class NetworkPortData(BaseModel):
    SEND: PortMapping
    RECV: PortMapping
