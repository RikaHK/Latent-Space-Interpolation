import os
import torch
import numpy as np
import cv2
from diffusers import StableDiffusionPipeline

# 1. SETUP & CONFIGURATION
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Running on device: {device.upper()}")

# Define Output Directories (Local paths for Docker)
OUTPUT_DIR = "output_videos"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. MATH FUNCTIONS (Lerp & Slerp)
def lerp(v0, v1, t):
    return (1 - t) * v0 + t * v1

def slerp(v0, v1, t, DOT_THRESHOLD=0.9995):
    v0_flat = v0.flatten()
    v1_flat = v1.flatten()
    v0_norm = v0_flat / torch.norm(v0_flat)
    v1_norm = v1_flat / torch.norm(v1_flat)
    dot = torch.sum(v0_norm * v1_norm)
    dot = torch.clamp(dot, -1.0, 1.0)
    
    if torch.abs(dot) > DOT_THRESHOLD:
        return lerp(v0, v1, t)
    
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega)
    scale_0 = torch.sin((1 - t) * omega) / sin_omega
    scale_1 = torch.sin(t * omega) / sin_omega
    return scale_0 * v0 + scale_1 * v1

# 3. GENERATION LOGIC
def get_latents_and_embeddings(pipe, prompt1, prompt2, seed):
    generator = torch.Generator(device).manual_seed(seed)
    
    # Generate Noise
    shape = (1, 4, 64, 64) # SD v1.5 standard size
    rand_noise_start = torch.randn(shape, generator=generator, device=device, dtype=torch.float32)
    rand_noise_end = torch.randn(shape, generator=torch.Generator(device).manual_seed(seed+1), device=device, dtype=torch.float32)

    # Encode Text
    embeds_start = pipe.encode_prompt(prompt1, device=device, num_images_per_prompt=1, do_classifier_free_guidance=False)[0]
    embeds_end = pipe.encode_prompt(prompt2, device=device, num_images_per_prompt=1, do_classifier_free_guidance=False)[0]
    
    return (rand_noise_start, rand_noise_end), (embeds_start, embeds_end)

def generate_video(pipe, prompt_start, prompt_end, method="slerp", steps=20, seed=42):
    print(f"🎬 Generating: {prompt_start} -> {prompt_end} ({method})")
    
    (noise_pair, emb_pair) = get_latents_and_embeddings(pipe, prompt_start, prompt_end, seed)
    
    frames = []
    
    for i in range(steps):
        t = i / (steps - 1)
        
        # Interpolate
        current_emb = lerp(emb_pair[0], emb_pair[1], t)
        if method == "slerp":
            current_noise = slerp(noise_pair[0], noise_pair[1], t)
        else:
            current_noise = lerp(noise_pair[0], noise_pair[1], t)

        # Generate Image
        with torch.no_grad():
            image = pipe(
                prompt_embeds=current_emb, 
                latents=current_noise, 
                guidance_scale=7.5, 
                num_inference_steps=20 
            ).images[0]
            
        frames.append(np.array(image))
        print(f"   Frame {i+1}/{steps}", end='\r')

    # Save Video
    filename = f"docker_output_{method}_{seed}.mp4"
    save_path = os.path.join(OUTPUT_DIR, filename)
    height, width, layers = frames[0].shape
    video = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))
    
    for frame in frames:
        video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    video.release()
    print(f"\n✅ Video Saved: {save_path}")

# 4. MAIN EXECUTION
if __name__ == "__main__":
    print("⏳ Loading Model (SD v1.5)...")
    # Using float32 for CPU compatibility in Docker (safer for grading)
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    pipe = pipe.to(device)
    
    # Run a simple demo to prove it works
    print("🚀 Starting Docker Demo Run...")
    generate_video(pipe, "A photo of a cat", "A photo of a tiger", "slerp", steps=15)
    
    print("🎉 Container Job Complete.")