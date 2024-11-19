import random
import time
import gradio as gr
import cv2
import tempfile
from ultralytics import YOLOv10

global model
# model = YOLOv10.from_pretrained(f'jameslahm/yolov10m')
model = YOLOv10('best.pt')

def yolov10_inference(image, video, model_id, image_size, conf_threshold):
    # model = YOLOv10.from_pretrained(f'jameslahm/{model_id}')
    # if random.randrange(11) > 9:
    #     return image,None
    
    if image:
        results = model.predict(source=image, imgsz=image_size, conf=conf_threshold,classes=[0,1],save=True,save_txt=True,name="test",show_labels=False,show_boxes=False)
        annotated_image = results[0].plot()
        return annotated_image[:, :, ::-1], None
    else:
        video_path = tempfile.mktemp(suffix=".webm")
        with open(video_path, "wb") as f:
            with open(video, "rb") as g:
                f.write(g.read())

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_video_path = tempfile.mktemp(suffix=".webm")
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'vp80'), fps, (frame_width, frame_height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(source=frame, imgsz=image_size, conf=conf_threshold)
            annotated_frame = results[0].plot()
            out.write(annotated_frame)

        cap.release()
        out.release()

        return None, output_video_path


def yolov10_inference_for_examples(image, model_path, image_size, conf_threshold):
    annotated_image, _ = yolov10_inference(image, None, model_path, image_size, conf_threshold)
    return annotated_image

def app():
    with gr.Blocks():
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="pil", label="Image", visible=True,streaming=True)
                video = gr.Video(label="Video", visible=False)
                input_type = gr.Radio(
                    choices=["Image", "Video"],
                    value="Image",
                    label="Input Type",
                    visible=False,
                )
                model_id = gr.Dropdown(
                    label="Model",
                    choices=[
                        "yolov10n",
                        "yolov10s",
                        "yolov10m",
                        "yolov10b",
                        "yolov10l",
                        "yolov10x",
                    ],
                    value="yolov10m",
                    visible=False,
                )
                image_size = gr.Slider(
                    label="Image Size",
                    minimum=320,
                    maximum=1280,
                    step=32,
                    value=640,
                )
                conf_threshold = gr.Slider(
                    label="Confidence Threshold",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    value=0.65,
                )
                yolov10_infer = gr.Button(
                    value="Detect Objects",
                    visible=False,
                )

            with gr.Column():
                output_image = gr.Image(type="numpy", label="Annotated Image", visible=True)
                output_video = gr.Video(label="Annotated Video", visible=False)

        def update_visibility(input_type):
            image = gr.update(visible=True) if input_type == "Image" else gr.update(visible=False)
            video = gr.update(visible=False) if input_type == "Image" else gr.update(visible=True)
            output_image = gr.update(visible=True) if input_type == "Image" else gr.update(visible=False)
            output_video = gr.update(visible=False) if input_type == "Image" else gr.update(visible=True)

            return image, video, output_image, output_video

        input_type.change(
            fn=update_visibility,
            inputs=[input_type],
            outputs=[image, video, output_image, output_video],
        )

        def run_inference(image, video, model_id, image_size, conf_threshold, input_type):
            if input_type == "Image":
                return yolov10_inference(image, None, model_id, image_size, conf_threshold)
            else:
                return yolov10_inference(None, video, model_id, image_size, conf_threshold)

        yolov10_infer.click(
            fn=run_inference,
            inputs=[image, video, model_id, image_size, conf_threshold, input_type],
            outputs=[output_image, output_video],
            trigger_mode="always_last",
            show_progress=False
        )

        image.change(
            fn=run_inference,
                    inputs=[image, video, model_id, image_size, conf_threshold, input_type],
                    outputs=[output_image, output_video],
                    trigger_mode='once',
                    queue=False,
                    show_progress=False
                    )

shortcut_js = """
<script>
        var intervalID = setInterval(function() {
        //document.getElementById('component-14').click();
        console.log('Auto click')
        }, 30000);
</script>
"""

gradio_app = gr.Blocks(head=shortcut_js,title="錄影中 請微笑")
with gradio_app:
    gr.HTML(
        """
    <h1 style='text-align: center'>
    錄影中 請微笑
    </h1>
    """)
    gr.HTML(
        """
        <h3 style='text-align: center'>
        <a></a>
        </h3>
        """)
    with gr.Row():
        with gr.Column():
            app()
if __name__ == '__main__':
    gradio_app.launch(server_port=80,server_name="0.0.0.0")
