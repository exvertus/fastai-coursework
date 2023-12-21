from fastai.vision.all import load_learner, PILImage
from pathlib import Path
import gradio

here = Path(__file__).parent
predicter = load_learner(here / 'export.pkl')

labels = predicter.dls.vocab

def predict(img):
    img = PILImage.create(img)
    pred,inx,probs = predicter.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

example_dir = here / 'test'
gradio.Interface(
    fn=predict,
    inputs=gradio.Image(),
    outputs=gradio.Label(num_top_classes=3),
    title="Guitar detector",
    description="Is it a Stratocaster, Telecaster, SG, or Les Paul?<br>Resnet18 trained on ~500 images: ~93% accurate.<br>The last Stratocaster example demos some of its inaccuracy.",
    article="<p style='text-align: center'><a href='https://github.com/exvertus/fastai-coursework/tree/main/two/guitars' target='_blank'>See the code</a></p>",
    examples=[
        example_dir / 'strat.jpg', 
        example_dir / 'tele.jpg', 
        example_dir / 'sg.jpg', 
        example_dir / 'lespaul.jpg',
        example_dir / 'mistake-example.jpg']
    ).launch(share=True)
