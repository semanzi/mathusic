import math
from scamp import Session
from scamp_extensions.pitch import Scale

session = Session()
instrument = session.new_part("Piano")

# Create a natural minor scale rooted at MIDI pitch 60 (Middle C)
musical_scale = Scale.natural_minor(60)

for beat in range(50):
    # Smoothly calculating a sine wave value over time
    raw_pitch = 65 + (12 * math.sin(beat * 0.4))
    
    # FIX: Use .round() to snap the raw float to the nearest scale note
    scale_pitch = musical_scale.round(raw_pitch)
    
    # Play the note (Pitch, Volume 0 to 1, Duration in beats)
    instrument.play_note(scale_pitch, 0.7, 0.5)

