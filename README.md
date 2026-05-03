# situated-vision
Probing CLIP's cultural situatedness through 993 dance-related objects from the Pitt Rivers Museum online collection.

> *"The knowing self is partial in all its guises."* — Donna Haraway

## Research Question
How does CLIP's visual understanding diverge from the archival logic of colonial museum collections — and what does this reveal about the cultural situatedness of CLIP?

## Dataset
993 dance-related objects from the [Pitt Rivers Museum Collections Online](https://www.prm.ox.ac.uk/collections-online), University of Oxford. Initial selection included 1,000 objects. 7 were excluded: no images available.

## Data Access 
Images were retrieved via the IIIF image server at dams.prm.ox.ac.uk. IIIF is an open standard designed for machine access to cultural heritage collections. Please respect the Pitt Rivers Museum's terms of use and rate limits when using this code. Images or outputs including images are not included in this repository. 

## Method
**1. Data Collection**
Downloaded 1000 dance-related objects from Pitt Rivers Museum Collections Online as CSV. Retrieved images by reverse-engineering the IIIF image server (`dams.prm.ox.ac.uk`) — each accession number maps directly to a IIIF manifest containing the image URL.

**2. Image Embeddings**
Passed 993 images through CLIP (ViT-B/32) image encoder. Each image becomes a 512-dimensional vector representing its position in CLIP's semantic space.

**3. Text Embeddings**
Encoded each object's museum description through CLIP's text encoder. Same 512-dimensional space — allowing direct comparison with image vectors.

**4. Dimensionality Reduction**
Applied UMAP to reduce 512D vectors to 2D and 3D for visualization. Run separately on image embeddings, text embeddings, and both combined.

**5. Gap Analysis**
Computed cosine similarity between each object's image and text embedding. Low similarity = large gap between what CLIP sees and what the museum says.

**6. Zero-Shot Retrieval**
Used CLIP's text encoder to query the image embeddings with natural language prompts (e.g. "face", "death and ritual"). Results reveal how CLIP's visual ontology maps onto — or diverges from — museum categories.

## Key Findings
- CLIP clusters objects by visual form, not cultural origin — masks from Africa, Oceania and Mexico group together regardless of provenance
- Image and text embeddings occupy fundamentally separate semantic spaces
- The gap is largest where museum language uses cultural interpretation ("devil-dances") rather than visual description
- Zero-shot queries reveal CLIP's cultural bias directly: "dance" returns Western ballet figures as top results despite the dataset consisting entirely of non-Western dance objects; "fear" finds grimacing masks; "love" finds red shoes and fans

## Outputs
UMAP scatter plots (regional, 3D), image grids, zero-shot retrieval grids, interactive HTML visualizations (run 09_umap_3d.py and 11_text_image_umap.py locally). Images not included — run the pipeline locally.

## Stack
Python · CLIP · UMAP · PyTorch · rembg · plotly · pandas

## Note 
Parts of the code were generated with AI assistance (Claude, Sonnet 4.6, Anthropic).


