import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nli_checker import check_nli

claim = "NASA's Apollo 11 mission landed the first humans on the Moon on July 20, 1969."

p_spaceref = "On July 20, 1969, Apollo 11 astronauts Armstrong and Aldrin landed in the Sea of Tranquility, where Armstrong famously stepped onto the lunar surface, marking the first time humans set foot on another celestial body."
print("Spaceref:", check_nli(claim, p_spaceref))

p_nasa_full = (
    "TITLE: July 20, 1969: One Giant Leap For Mankind - NASA\n\n"
    "Apollo 11 astronauts Neil Armstrong, Buzz Aldrin and Michael Collins launched to the Moon. "
    "When the lunar module lands at 4:17 p.m EDT, Armstrong radios Houston, Tranquility Base here. The Eagle has landed. "
    "At 10:56 p.m. EDT Armstrong is ready to plant the first human foot on another world."
)
print("NASA full window:", check_nli(claim, p_nasa_full))
