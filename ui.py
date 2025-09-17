# ui.py
# This module builds and runs the Gradio web app for dominant color extraction.
# It provides interfaces for both image and video analysis using the dominant_colors package.

import gradio as gr

def build_demo(process_image, process_video):
    demo_title = "Dominant Colors – Image & Video"
    demo_description = (
        "# Dominant Colors – Image & Video\n"
        "Upload an image **or** a video to extract the top dominant colors. "
        "You will receive both a palette visualization and a table of HEX/RGB values with percentages."
    )

    with gr.Blocks(title=demo_title, theme=gr.themes.Soft()) as demo:
        gr.Markdown(demo_description)

        with gr.Tabs():
            # Image analysis tab
            with gr.Tab("Image"):
                with gr.Row():
                    image_in = gr.Image(type="filepath", label="Upload image", sources=["upload"], height=300)
                with gr.Accordion("Parameters", open=False):
                    k_img = gr.Slider(1, 12, value=5, step=1, label="Top K colors")
                    sample_img = gr.Slider(100, 1024, value=400, step=10, label="Downsample max side (px)")
                    excl_img = gr.Checkbox(value=True, label="Exclude near-black/near-white")
                    labels_img = gr.Checkbox(value=True, label="Show labels on palette")
                run_img = gr.Button("Analyze Image")
                palette_img = gr.Image(type="filepath", label="Palette", interactive=False)
                table_img = gr.Dataframe(headers=["hex", "rgb", "percent"], label="Colors", interactive=False)

                run_img.click(
                    fn=process_image,
                    inputs=[image_in, k_img, sample_img, excl_img, labels_img],
                    outputs=[palette_img, table_img],
                    concurrency_limit=2,  # remove if your Gradio complains
                )

            # Video analysis tab
            with gr.Tab("Video"):
                with gr.Row():
                    video_in = gr.Video(label="Upload video", sources=["upload"], height=300)
                with gr.Accordion("Parameters", open=False):
                    k_vid = gr.Slider(1, 12, value=5, step=1, label="Top K colors")
                    step_vid = gr.Slider(1, 60, value=15, step=1, label="Frame step (every Nth frame)")
                    resize_vid = gr.Slider(64, 720, value=256, step=16, label="Downsample max side (px)")
                    limit_vid = gr.Slider(10, 1000, value=300, step=10, label="Max sampled frames")
                    excl_vid = gr.Checkbox(value=True, label="Exclude near-black/near-white")
                    labels_vid = gr.Checkbox(value=True, label="Show labels on palette")
                run_vid = gr.Button("Analyze Video")
                palette_vid = gr.Image(type="filepath", label="Palette", interactive=False)
                table_vid = gr.Dataframe(headers=["hex", "rgb", "percent"], label="Colors", interactive=False)

                run_vid.click(
                    fn=process_video,
                    inputs=[video_in, k_vid, step_vid, resize_vid, limit_vid, excl_vid, labels_vid],
                    outputs=[palette_vid, table_vid],
                    concurrency_limit=2,  # remove if your Gradio complains
                )

        gr.Markdown(
            """
**Notes**
- Results depend on K, sampling, and whether extremes are excluded.
- Video analysis uses frame sampling for a global palette; no per-frame breakdown.
- Processing runs on CPU and is suitable for free hosting tiers.
"""
        )

    return demo
