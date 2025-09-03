#!/usr/bin/env python3
"""Debug Pydantic schema generation"""

import json
from enum import StrEnum
from pydantic import BaseModel, Field


class TemperatureUnit(StrEnum):
    CELSIUS = "celsius"
    FAHRENHEIT = "fahrenheit"


class GetWeather(BaseModel):
    """Get the weather at the given location in the given temperature unit of measurement"""

    location: str = Field(description="The location to get the weather for")
    unit: TemperatureUnit = Field(description="Temperature Unit", default=TemperatureUnit.CELSIUS)


if __name__ == "__main__":
    schema = GetWeather.model_json_schema()
    print("Pydantic schema:")
    print(json.dumps(schema, indent=2))