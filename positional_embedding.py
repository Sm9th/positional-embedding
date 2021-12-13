import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from pytorch_pretrained_vit import ViT
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device = ", device)

model = ViT('B_16_imagenet1k', pretrained=True)
model.fc = nn.Linear(768,37)
# the model is uploaded to github
model.load_state_dict(torch.load('hw3_1_model.pt'))

pos_embed = model.positional_embedding.pos_embedding

cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
fig = plt.figure(figsize=(10, 10))

for i in range(1, pos_embed.shape[1]):
    sim = F.cosine_similarity(pos_embed[0, i:i+1], pos_embed[0, 1:], dim=1)
    sim = sim.reshape((24, 24)).detach().cpu().numpy()
    ax = fig.add_subplot(24, 24, i)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.imshow(sim)

plt.savefig('figpng')
