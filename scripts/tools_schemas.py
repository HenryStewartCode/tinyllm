WEATHER = {"type": "object", "properties": {"city": {"type": "string", "description": "The city, e.g. Paris"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city", "unit"], "additionalProperties": False}
TIME = {"type": "object", "properties": {"timezone": {"type": "string", "description": "IANA timezone, e.g. Europe/Paris"}}, "required": ["timezone"], "additionalProperties": False}
FLIGHT = {"type": "object", "properties": {"origin": {"type": "string"}, "destination": {"type": "string"}, "passengers": {"type": "integer", "minimum": 1, "maximum": 9}, "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 3}}, "required": ["origin", "destination", "passengers"], "additionalProperties": False}
def tool(name, description, parameters, **extra):
    return {"type": "function", "function": {"name": name, "description": description, "parameters": parameters, **extra}}
TOOLS = [
    tool("get_weather", "Get the current weather in a city", WEATHER),
    tool("get_time", "Get the current time in a timezone", TIME),
    tool("search_flights", "Search for flights between two cities", FLIGHT),
]
SCHEMAS = {t["function"]["name"]: t["function"]["parameters"] for t in TOOLS}
ASKS_FOR_TOOLS = [
    "What's the weather like in Paris right now, in celsius?",
    "What time is it in Tokyo?",
    "Find me flights from London to New York for 2 passengers.",
    "How hot is it in Cairo today? Use fahrenheit.",
    "I need to know the current time in Europe/Berlin.",
    "Please check the weather in Sydney in celsius.",
    "Book search: flights from Rome to Madrid, 1 passenger.",
    "What is the temperature in Toronto in fahrenheit?",
]

ASKS = ASKS_FOR_TOOLS
