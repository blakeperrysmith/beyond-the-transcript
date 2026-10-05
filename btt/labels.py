"""Label set and corpus metadata.

The classes are the intersection of what CREMA-D and RAVDESS label. RAVDESS
"calm" and "surprised" have no CREMA-D counterpart and are excluded rather than
merged into a neighbour.
"""

CLASSES = ["neutral", "happy", "sad", "angry", "fearful", "disgust"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}

# RAVDESS filename field 3.
RAVDESS_EMOTION = {
    "01": "neutral",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
}  # 02 calm and 08 surprised are dropped on purpose

# CREMA-D filename field 3.
CREMAD_EMOTION = {
    "NEU": "neutral",
    "HAP": "happy",
    "SAD": "sad",
    "ANG": "angry",
    "FEA": "fearful",
    "DIS": "disgust",
}

# Sentence text as documented by the datasets' authors.
RAVDESS_STATEMENT = {
    "01": "Kids are talking by the door",
    "02": "Dogs are sitting by the door",
}
CREMAD_SENTENCE = {
    "IEO": "It's eleven o'clock",
    "TIE": "That is exactly what happened",
    "IOM": "I'm on my way to the meeting",
    "IWW": "I wonder what this is about",
    "TAI": "The airplane is almost full",
    "MTI": "Maybe tomorrow it will be cold",
    "IWL": "I would like a new alarm clock",
    "ITH": "I think I have a doctor's appointment",
    "DFA": "Don't forget a jacket",
    "ITS": "I think I've seen this before",
    "TSI": "The surface is slick",
    "WSI": "We'll stop in a couple of minutes",
}

LICENSES = {
    "cremad": "CREMA-D, Cao et al. 2014, Open Database License (ODbL)",
    "ravdess": "RAVDESS, Livingstone and Russo 2018, CC BY-NC-SA 4.0 (non-commercial)",
}
