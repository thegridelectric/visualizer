import math
import time
from typing import List, Literal, Optional
from aiohttp import web
from homeassistant.components.http import HomeAssistantView
from pydantic import BaseModel, PositiveInt, field_validator

R_FIXED_KOHMS = 5.65  # The voltage divider resistors in the TankModule
THERMISTOR_T0 = 298  # i.e. 25 degrees
THERMISTOR_R0_KOHMS = 10  # The R0 of the NTC thermistor - an industry standard
THERMISTOR_BETA = 3977


class MicroVolts(BaseModel):
    HwUid: str
    AboutNodeNameList: List[str]
    MicroVoltsList: List[int]
    TypeName: Literal["microvolts"] = "microvolts"
    Version: Literal["100"] = "100"


class TankModuleParams(BaseModel):
    HwUid: str
    ActorNodeName: str
    PicoAB: str
    CapturePeriodS: PositiveInt
    Samples: PositiveInt
    NumSampleAverages: PositiveInt
    AsyncCaptureDeltaMicroVolts: PositiveInt
    CaptureOffsetS: Optional[float] = None
    TypeName: Literal["tank.module.params"] = "tank.module.params"
    Version: Literal["100"] = "100"

    @field_validator("PicoAB")
    @classmethod
    def check_pico_a_b(cls, v: str) -> str:
        if v not in {"a", "b"}:
            raise ValueError("PicoAB must be 'a' or 'b'")
        return v


class ApiTankModule(HomeAssistantView):
    url = "/api/{subpath}"
    name = "api_tank_module"
    requires_auth = True

    def __init__(self, hass):
        self.hass = hass
        self.actor_node_name = 'unknown'

    async def post(self, request: web.Request, subpath: str):
        try:
            self.actor_node_name = subpath.split('-')[0]
            subpath_cropped = subpath.replace(f'{self.actor_node_name}-', '')
        except Exception as e:
            return web.json_response({"error": "Invalid subpath"}, status=400)
        try:
            data = await request.json()
        except Exception as e:
            return web.json_response({"error": "Invalid JSON"}, status=400)
        if subpath_cropped == "tank-module-params":
            return await self._handle_params_post(data)
        elif subpath_cropped == "microvolts":
            return await self._handle_microvolts_post(data)
        else:
            return web.json_response({"error": f"Unknown subpath: {subpath}"}, status=404)

    async def _handle_params_post(self, data: dict) -> web.Response:
        try:
            params = TankModuleParams(**data)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)
        if params.ActorNodeName != self.actor_node_name:
            return web.json_response({"error": "Incorrect ActorNodeName"}, status=400)
        # Return updated params
        period = params.CapturePeriodS
        offset = round(period - time.time() % period, 3) - 2
        new_params = TankModuleParams(
            HwUid=params.HwUid,
            ActorNodeName=self.actor_node_name,
            PicoAB=params.PicoAB,
            CapturePeriodS=params.CapturePeriodS,
            Samples=params.Samples,
            NumSampleAverages=params.NumSampleAverages,
            AsyncCaptureDeltaMicroVolts=params.AsyncCaptureDeltaMicroVolts,
            CaptureOffsetS=offset,
        )
        return web.json_response(new_params.model_dump())

    async def _handle_microvolts_post(self, data: dict) -> web.Response:
        try:
            micro = MicroVolts(**data)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)
        responses = []
        for i, value in enumerate(micro.MicroVoltsList):
            name = micro.AboutNodeNameList[i].replace('-','_')
            try:
                volts = value / 1e6
                temp_c = round(self.simple_beta_for_pico(volts), 2)
                entity_id = f"sensor.{name.lower()}_temperature"
                self.hass.states.async_set(
                    entity_id,
                    temp_c,
                    {
                        "unit_of_measurement": "°C",
                        "friendly_name": f"{name} Temperature",
                        "source": micro.HwUid,
                    }
                )
                responses.append({entity_id: temp_c})
            except Exception as e:
                responses.append({name: f"error: {str(e)}"})
        return web.json_response({"updated": responses})

    def simple_beta_for_pico(self, volts: float, fahrenheit=False) -> float:
        if volts <= 0 or volts >= 3.3:
            raise ValueError(f"Invalid voltage ({volts} V) for thermistor calculation")
        r_fixed = R_FIXED_KOHMS
        r_therm = 1 / ((3.3 / volts - 1) / r_fixed)
        if r_therm <= 0:
            self.hass.logger.warning("Disconnected thermistor!")
        t0, r0 = THERMISTOR_T0, THERMISTOR_R0_KOHMS
        beta = THERMISTOR_BETA
        temp_c = 1 / ((1 / t0) + (math.log(r_therm / r0) / beta)) - 273
        temp_f = 32 + (temp_c * 9 / 5)
        return round(temp_f, 2) if fahrenheit else round(temp_c, 2)
