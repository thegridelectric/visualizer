"""Type flo.params.house0, version 000"""
import uuid
import time
from typing import List, Literal, Optional
from pydantic import BaseModel, PositiveInt, StrictInt, Field

class FloParamsHouse0(BaseModel):
    GNodeAlias: str
    FloParamsUid: str = Field(default_factory=lambda: str(uuid.uuid4()))
    TimezoneStr: str = "America/New_York"
    StartUnixS: int
    HorizonHours: PositiveInt = 48
    NumLayers: PositiveInt = 24
    # Equipment
    StorageVolumeGallons: PositiveInt = 360
    StorageLossesPercent: float = 0.5
    HpMinElecKw: float = -0.5
    HpMaxElecKw: float = 11
    CopIntercept: float = 1.02
    CopOatCoeff: float = 0.0257
    CopLwtCoeff: float = 0
    CopMin: float = 1.4
    CopMinOatF: float = 15
    HpTurnOnMinutes: int = 10
    # Initial state
    InitialTopTempF: StrictInt
    InitialMiddleTempF: Optional[StrictInt] = 0
    InitialBottomTempF: StrictInt
    InitialThermocline: Optional[StrictInt] = 12
    InitialThermocline1: Optional[StrictInt] = 12
    InitialThermocline2: Optional[StrictInt] = 12
    HpIsOff: bool = False
    BufferAvailableKwh: float = 0
    HouseAvailableKwh: float = 0
    # Forecasts
    LmpForecast: Optional[List[float]] = None
    DistPriceForecast: Optional[List[float]] = None
    RegPriceForecast: Optional[List[float]] = None
    PriceForecastUid: str
    OatForecastF: Optional[List[float]] = None
    WindSpeedForecastMph: Optional[List[float]] = None
    WeatherUid: str
    # House parameters
    AlphaTimes10: StrictInt
    BetaTimes100: StrictInt
    GammaEx6: StrictInt
    IntermediatePowerKw: float
    IntermediateRswtF: StrictInt
    DdPowerKw: float
    DdRswtF: StrictInt
    DdDeltaTF: StrictInt
    MaxEwtF: StrictInt
    PriceUnit: str
    ParamsGeneratedS: int
    FloAlias: str = "Winter.Oak"
    FloGitCommit: str = "6828dea"
    TypeName: Literal["flo.params.house0"] = "flo.params.house0"
    Version: Literal["000", "001", "002", "003"] = "003"

    def to_dict(self):
        return vars(self)