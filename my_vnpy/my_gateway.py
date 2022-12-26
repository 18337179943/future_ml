from vnpy.event import Event
from vnpy_ctp import CtpGateway as OriginalGateway
from vnpy_ctp import CtpTdApi as OriginalTdApi
# from vnpy.gateway.ctp.ctp_gateway import CtpGateway as OriginalGateway
# from vnpy.gateway.ctp.ctp_gateway import CtpTdApi as OriginalTdApi

from .my_event import EVENT_GATEWAY_READY


class CtpGateway(OriginalGateway):
    """"""
    
    def __init__(self, event_engine):
        """Constructor"""
        super().__init__(event_engine)

        self.td_api = CtpTdApi(self)

class CtpTdApi(OriginalTdApi):
    
    def onRspQryInstrument(self, data: dict, error: dict, reqid: int, last: bool):
        """
        Callback of instrument query.
        """
        super().onRspQryInstrument(data, error, reqid, last)

        if last:
            self.gateway.on_event(EVENT_GATEWAY_READY, self.gateway_name)
