from fastai.vision.all import load_learner, PILImage
from pathlib import Path
import gradio

here = Path(__file__).parent
predicter = load_learner(here / 'seasons.pkl')

labels = predicter.dls.vocab

def predict(img):
    _img = PILImage.create(img)
    _, _, probabilities = predicter.predict(_img)
    return {labels[i]: float(prob) for i, prob in enumerate(probabilities)}

example_dir = here / 'test'
gradio.Interface(
    fn=predict,
    inputs=gradio.Image(),
    outputs=gradio.Label(num_top_classes=3),
    title='Season Guesser',
    description='What season is this image?',
    article="<p style='text-align: center'><a href='https://github.com/exvertus/fastai-coursework/tree/main/one/seasons' target='_blank'>See the code</a></p>",
    examples=[
        example_dir / 'summer.jpg',
        example_dir / 'winter.jpg',
        example_dir / 'autumn.jpg',
        example_dir / 'winter-art.jpg',
    ]
).launch()
