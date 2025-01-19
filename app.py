import gradio as gr
import torch
import os
from kokoro import generate
from models import build_model

# Initialize model and device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL = build_model('kokoro-v0_19.pth', device)

# Load the voice models
voices = {
    'af': torch.load("voices/af.pt", weights_only=True),
    'af_bella': torch.load("voices/af_bella.pt", weights_only=True),
    'af_sarah': torch.load("voices/af_sarah.pt", weights_only=True),
    'am_adam': torch.load("voices/am_adam.pt", weights_only=True),
    'am_michael': torch.load("voices/am_michael.pt", weights_only=True),
    'bf_emma': torch.load("voices/bf_emma.pt", weights_only=True),
    'bf_isabella': torch.load("voices/bf_isabella.pt", weights_only=True),
    'bm_george': torch.load("voices/bm_george.pt", weights_only=True),
    'bm_lewis': torch.load("voices/bm_lewis.pt", weights_only=True),
    'af_nicole': torch.load("voices/af_nicole.pt", weights_only=True),
    'af_sky': torch.load("voices/af_sky.pt", weights_only=True)
}


def parse_voice_formula(formula):
    """Parse the voice formula string and return the combined voice tensor."""
    if not formula.strip():
        raise ValueError("Empty voice formula")
        
    # Initialize the weighted sum
    weighted_sum = None
    
    # Split the formula into terms
    terms = formula.split('+')
    
    for term in terms:
        # Parse each term (format: "0.333 * voice_name")
        weight, voice_name = term.strip().split('*')
        weight = float(weight.strip())
        voice_name = voice_name.strip()
        
        # Get the voice tensor
        if voice_name not in voices:
            raise ValueError(f"Unknown voice: {voice_name}")
            
        voice_tensor = voices[voice_name]
        
        # Add to weighted sum
        if weighted_sum is None:
            weighted_sum = weight * voice_tensor
        else:
            weighted_sum += weight * voice_tensor
            
    return weighted_sum

def get_new_voice(formula):
    try:
        # Parse the formula and get the combined voice tensor
        weighted_voices = parse_voice_formula(formula)
        
        # Save and load the combined voice
        torch.save(weighted_voices, "weighted_normalised_voices.pt")
        VOICEPACK = torch.load("weighted_normalised_voices.pt", weights_only=False).to(device)
        return VOICEPACK
    except Exception as e:
        raise gr.Error(f"Failed to create voice: {str(e)}")

def text_to_speech(text, formula):
    try:
        if not text.strip():
            raise gr.Error("Please enter some text")
            
        if not formula.strip():
            raise gr.Error("Please select at least one voice")
            
        # Get the combined voice
        VOICEPACK = get_new_voice(formula)
        
        # Generate audio
        audio, phonemes = generate(MODEL, text, VOICEPACK, lang='a')
        return (24000, audio)
    except Exception as e:
        raise gr.Error(f"Failed to generate speech: {str(e)}")
        

custom_css = """
/* Main title */
.heading {
    color: rgb(76, 175, 147) !important;
    font-size: 2em !important;
    font-weight: 600 !important;
    text-align: center !important;
    margin: 20px 0 10px 0 !important;
    width: 100% !important;
}
/* Description text - Dark mode */
.description {
    color: var(--body-text-color, rgba(76, 175, 147, 0.7)) !important;
    text-align: center !important;
    max-width: 800px !important;
    margin: 0 auto 30px auto !important;
    font-size: 0.9em !important;
    line-height: 1.6 !important;
}
/* Description text - Light mode override */
.light-mode .description, 
[data-theme="light"] .description {
    color: rgba(55, 65, 81, 0.9) !important;
}
/* Description text - Bold elements in light mode */
.light-mode .description b, 
[data-theme="light"] .description b {
    color: rgb(55, 65, 81) !important;
    font-weight: 600 !important;
}
.container-wrap {
    display: flex !important;
    gap: 5px !important;
    justify-content: center !important;
    margin: 0 auto !important;
    max-width: 1400px !important;  /* Increased max-width */
}
.vert-group {
    min-width: 100px !important;   /* Increased from 80px */
    width: 105px !important;       /* Increased from 90px */
    flex: 0 0 auto !important;
}
.vert-group label {
    white-space: nowrap !important;
    overflow: visible !important;
    width: auto !important;
    font-size: 0.85em !important;  /* Slightly increased font size */
    transform-origin: left center !important;
    transform: rotate(0deg) translateX(-50%) !important;
    position: relative !important;
    left: 50% !important;
    display: inline-block !important;
    text-align: center !important;
    margin-bottom: 5px !important;
    padding: 0 1px !important;     /* Added padding */
}
.vert-group .wrap label {
    text-align: center !important;
    width: 100% !important;
    display: block !important;
}
/* Hover effect */
.vert-group:hover {
    transform: translateY(-5px) !important;
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2) !important;
}
.slider_input_container {
    height: 200px !important;
    position: relative !important;
    width: 50px !important;        /* Increased from 40px */
    margin: 0 auto !important;
    overflow: hidden !important;
}
.slider_input_container input[type="range"] {
    position: absolute !important;
    width: 200px !important;
    left: -75px !important;        /* Adjusted from -80px */
    top: 100px !important;
    transform: rotate(90deg) !important;
}
.min_value {
    position: absolute !important;
    bottom: 0 !important;
    left: 10px !important;
}
.max_value {
    position: absolute !important;
    top: 0 !important;
    left: 10px !important;
}
.tab-like-container {
    transform: scale(0.8) !important;
}
.gradio-row, .gradio-column {
    background: none !important;
    border: none !important;
    min-width: unset !important;
}
.heading {
    text-align: center !important;
    margin-bottom: 1rem !important;
}
.description {
    text-align: center !important;
    margin-bottom: 2rem !important;
    color: rgba(255, 255, 255, 0.7) !important;
}
/* Generate button */
#generate-btn {
    background: linear-gradient(90deg, rgb(76, 175, 147), rgb(76, 147, 175)) !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 12px 24px !important;
    color: white !important;
    font-weight: 600 !important;
    transition: transform 0.2s, box-shadow 0.2s !important;
}
#generate-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 5px 15px rgba(76, 175, 147, 0.3) !important;
}
"""

with gr.Blocks(css=custom_css, theme="ocean") as demo:
    gr.HTML(
        """
        <div class="heading">🎙️ AI Voice Mixer Studio - Kokoro TTS</div>
        <div class="description">
            <b>Mix and match different voices to create your perfect text-to-speech voice.<br>Each slider represents a 
            unique voice with distinct characteristics. This app lets you combine multiple voices with different weights 
            to create custom voice combinations. Select voices using checkboxes and adjust their weights using the sliders below!</b>
        </div>
        """
    )

    with gr.Row(variant="default", equal_height=True, elem_classes="container-wrap"):
        checkboxes = []
        sliders = []
        
        # Define slider configurations with emojis
        slider_configs = [
            ("af", "Default 👩‍🦰"),
            ("af_bella", "Bella 👩‍🦰🇺🇸"), 
            ("af_sarah", "Sarah 👩‍🦰🇺🇸"),
            ("af_nicole", "Nicole 👩‍🦰🇺🇸"), 
            ("af_sky", "Sky 👩‍🦰🇺🇸"),
            ("am_adam", "Adam 👨🇺🇸"),
            ("am_michael", "Michael 👨🇺🇸"),
            ("bf_emma", "Emma 👩‍🦰🇬🇧"),
            ("bf_isabella", "Isabella 👩‍🦰🇬🇧"),
            ("bm_george", "George 👨🇬🇧"),
            ("bm_lewis", "Lewis 👨🇬🇧")
        ]

        # Create columns for each slider
        for value, label in slider_configs:
            with gr.Column(min_width=70, scale=1, variant="default", elem_classes="vert-group"):
                checkbox = gr.Checkbox(label='')
                slider = gr.Slider(label=label, minimum=0, maximum=1, interactive=False, value=0, step=0.01)
                checkboxes.append(checkbox)
                sliders.append(slider)

    # Add voice combination formula display
    with gr.Row(equal_height=True):
        formula_display = gr.Textbox(
            label="Voice Combination Formula", 
            value="", 
            lines=2, 
            scale=4, 
            interactive=False,
            placeholder="This will begin to display immediately once any of the voice checkboxes is selected selected",
            info="Slider values are normalized to create this voice formula. Use the Sliders to intuitively increase or decrease a voice effect."
        )
        input_text = gr.Textbox(
            label="Input Text", 
            placeholder="Enter text to convert to speech", 
            lines=2, 
            scale=4
        )
        button_tts = gr.Button("🎙️ Generate Voice", scale=2, min_width=100, elem_id="generate-btn")

    # Generate speech from the selected custom voice
    with gr.Row(equal_height=True):
        kokoro_tts = gr.Audio(label="Generated Speech", type="numpy", autoplay=True)

    def generate_voice_formula(*values):
        """
        Generate a formatted string showing the normalized voice combination.
        Returns: String like "0.6 * voice1" or "0.4 * voice1 + 0.6 * voice2"
        """
        n = len(values) // 2
        checkbox_values = values[:n]
        slider_values = list(values[n:])

        # Get active sliders and their names
        active_pairs = [(slider_values[i], slider_configs[i][0])
                      for i in range(len(slider_configs))
                      if checkbox_values[i]]

        if not active_pairs:
            return ""

        # If only one voice is selected, use its actual value
        if len(active_pairs) == 1:
            value, name = active_pairs[0]
            return f"{value:.3f} * {name}"

        # Calculate sum for normalization of multiple voices
        total_sum = sum(value for value, _ in active_pairs)

        if total_sum == 0:
            return ""

        # Generate normalized formula for multiple voices
        terms = []
        for value, name in active_pairs:
            normalized_value = value / total_sum
            terms.append(f"{normalized_value:.3f} * {name}")

        return " + ".join(terms)

    def check_box(checkbox):
        """Handle checkbox changes."""
        if checkbox:
            return gr.Slider(interactive=True, value=1.0)  # Changed default to 1.0
        else:
            return gr.Slider(interactive=False, value=0)

    # Connect all checkboxes and sliders
    all_inputs = checkboxes + sliders

    # Update on checkbox changes
    for checkbox, slider in zip(checkboxes, sliders):
        checkbox.change(
            fn=check_box,
            inputs=[checkbox],
            outputs=[slider]
        )
        # Update formula on checkbox changes
        checkbox.change(
            fn=generate_voice_formula,
            inputs=all_inputs,
            outputs=[formula_display]
        )

    # Update formula on slider changes
    for slider in sliders:
        slider.change(
            fn=generate_voice_formula,
            inputs=all_inputs,
            outputs=[formula_display]
        )

    button_tts.click(
        fn=text_to_speech,
        inputs=[input_text, formula_display],
        outputs=[kokoro_tts]
    )


if __name__ == "__main__":
    demo.launch()