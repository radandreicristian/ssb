from src.image.fairface import FairFaceImageAttributeExtractor
import asyncio
import os.path as osp
import os

path = "data/concept_images/stabilityai-stable-diffusion-3/gang_activity"
extractor = FairFaceImageAttributeExtractor()

async def run():
    await asyncio.gather(
        [extractor.process_image(osp.join(path, p)) for p in os.listdir(path)]
    )

if __name__ == "__main__":
    asyncio.run(run)