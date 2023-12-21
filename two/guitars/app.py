from fastai.vision.all import load_learner, PILImage
from pathlib import Path
import gradio

here = Path(__file__).parent
predicter = load_learner(here / 'export.pkl')

labels = predicter.dls.vocab

print('break here')
def predict(img):
    img = PILImage.create(img)
    pred,inx,probs = predicter.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}


gradio.Interface(
    fn=predict, 
    inputs=gradio.Image(), 
    outputs=gradio.Label(num_top_classes=3)).launch(share=True)
