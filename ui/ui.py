import gradio as gr
from core import (
    change_label,
    change_tts_inference,
    open_asr,
    close_asr,
    open_denoise,
    close_denoise,
    open1a,
    close1a,
    open1b,
    close1b,
    open1c,
    close1c,
    open1abc,
    close1abc,
    open1Ba,
    close1Ba,
    open1Bb,
    close1Bb,
    open_slice,
    close_slice,
    switch_version,
    change_choices,
    sync,
    GPT_names,
    SoVITS_names,
    version,
    gpu_info,
    default_batch_size,
    default_max_batch_size,
    max_sovits_epoch,
    default_sovits_epoch,
    max_sovits_save_every_epoch,
    default_sovits_save_every_epoch,
    v3v4set,
    if_force_ckpt,
    default_batch_size_s1,
    gpus,
    pretrained_gpt_name,
    pretrained_sovits_name,
    n_cpu,
    precision_val,
    process_info,
    process_name_slice,
    process_name_denoise,
    process_name_asr,
    process_name_subfix,
    process_name_1a,
    process_name_1b,
    process_name_1c,
    process_name_1abc,
    process_name_sovits,
    process_name_gpt,
    process_name_tts,
    css,
    js,
)

def create_ui():
    with gr.Blocks(title="GPT-SoVITS WebUI", analytics_enabled=False, js=js, css=css) as app:

        with gr.Tabs():
            with gr.TabItem(("Dataset Preprocessing")):
                with gr.Accordion(("Speech Slicing")):
                    with gr.Row():
                        with gr.Column(scale=3):
                            with gr.Row():
                                slice_inp_path = gr.Textbox(label=("Speech slicer input (file or folder)"), value="")
                                slice_opt_root = gr.Textbox(
                                    label=("Speech slicer output folder"), value="output/sliced_speech"
                                )
                            with gr.Row():
                                threshold = gr.Textbox(
                                    label=("Noise gate threshold ( audio below this value will be treated as noise )"), value="-34"
                                )
                                min_length = gr.Textbox(
                                    label=("min_length: the minimum length of each segment. If the first segment is too short, it will be concatenated with the next segment until it exceeds this value"),
                                    value="4000",
                                )
                                min_interval = gr.Textbox(label=("Minumum interval for audio cutting"), value="300")
                                hop_size = gr.Textbox(
                                    label=("hop_size: FO hop size, the smaller the value, the higher the accuracy"),
                                    value="10",
                                )
                                max_sil_kept = gr.Textbox(label=("Maximum length for silence to be kept"), value="500")
                            with gr.Row():
                                _max = gr.Slider(
                                    minimum=0,
                                    maximum=1,
                                    step=0.05,
                                    label=("Loudness multiplier after normalization"),
                                    value=0.9,
                                    interactive=True,
                                )
                                alpha = gr.Slider(
                                    minimum=0,
                                    maximum=1,
                                    step=0.05,
                                    label=("alpha_mix: proportion of normalized audio merged into dataset"),
                                    value=0.25,
                                    interactive=True,
                                )
                            with gr.Row():
                                n_process = gr.Slider(
                                    minimum=1,
                                    maximum=n_cpu,
                                    step=1,
                                    label=("CPU threads used for audio slicing"),
                                    value=4,
                                    interactive=True,
                                )
                                slicer_info = gr.Textbox(label=process_info(process_name_slice, "info"))
                        open_slicer_button = gr.Button(
                            value=process_info(process_name_slice, "open"), variant="primary", visible=True
                        )
                        close_slicer_button = gr.Button(
                            value=process_info(process_name_slice, "close"), variant="primary", visible=False
                        )

                with gr.Row(visible=False):
                    with gr.Column(scale=3):
                        with gr.Row():
                            denoise_input_dir = gr.Textbox(label=("Input folder path"), value="")
                            denoise_output_dir = gr.Textbox(label=("Output folder path"), value="output/denoise_opt")
                        with gr.Row():
                            denoise_info = gr.Textbox(label=process_info(process_name_denoise, "info"))
                    open_denoise_button = gr.Button(
                        value=process_info(process_name_denoise, "open"), variant="primary", visible=True
                    )
                    close_denoise_button = gr.Button(
                        value=process_info(process_name_denoise, "close"), variant="primary", visible=False
                    )

                with gr.Accordion(("Speech Transcription")):
                    with gr.Row():
                        with gr.Column(scale=3):
                            with gr.Row():
                                asr_inp_dir = gr.Textbox(
                                    label=("Input folder path"), value="D:\\GPT-SoVITS\\raw\\xxx", interactive=True
                                )
                                asr_opt_dir = gr.Textbox(
                                    label=("Output folder path"), value="output/transcribed_audio", interactive=True
                                )
                            with gr.Row():
                                asr_model = gr.Dropdown(
                                    label=("ASR Model"),
                                    choices=["Faster Whisper (多语种)"],
                                    interactive=False,
                                    value="Faster Whisper (多语种)",
                                )
                                asr_size = gr.Dropdown(
                                    label=("ASR model size"),
                                    choices=['medium', 'medium.en', 'distil-large-v2', 'distil-large-v3', 'large-v1', 'large-v2', 'large-v3'],
                                    interactive=True,
                                    value="large",
                                )
                                asr_lang = gr.Dropdown(
                                    label=("ASR language"), choices=["en", "zh"], interactive=True, value="en"
                                )
                                asr_precision = gr.Dropdown(
                                    label=("Precision"),
                                    choices=["float16", "int8", "float32"],
                                    interactive=True,
                                    value=precision_val,
                                )
                            with gr.Row():
                                asr_info = gr.Textbox(label=process_info(process_name_asr, "info"))
                        open_asr_button = gr.Button(
                            value=process_info(process_name_asr, "open"), variant="primary", visible=True
                        )
                        close_asr_button = gr.Button(
                            value=process_info(process_name_asr, "close"), variant="primary", visible=False
                        )

                with gr.Accordion(("Transcription Proofreading")):
                    with gr.Row():
                        with gr.Column(scale=3):
                            with gr.Row():
                                path_list = gr.Textbox(
                                    label=("Label File Path (with file extension *.list)"),
                                    value="D:\\RVC1006\\GPT-SoVITS\\raw\\xxx.list",
                                    interactive=True,
                                )
                                label_info = gr.Textbox(label=process_info(process_name_subfix, "info"))
                        open_label = gr.Button(
                            value=process_info(process_name_subfix, "open"), variant="primary", visible=True
                        )
                        close_label = gr.Button(
                            value=process_info(process_name_subfix, "close"), variant="primary", visible=False
                        )

                    open_label.click(change_label, [path_list], [label_info, open_label, close_label])
                    close_label.click(change_label, [path_list], [label_info, open_label, close_label])

            with gr.TabItem(("Training")):
                with gr.Accordion(("Model Information")):
                    with gr.Row():
                        with gr.Row(equal_height=True):
                            exp_name = gr.Textbox(
                                label=("*Model Name"),
                                value="model_name",
                                interactive=True,
                                scale=3,
                            )
                            gpu_info_box = gr.Textbox(
                                label=("GPU Information"),
                                value=gpu_info,
                                visible=True,
                                interactive=False,
                                scale=5,
                            )
                            version_checkbox = gr.Radio(
                                label=("Pretrained Version"),
                                value=version,
                                choices=["v1", "v2", "v4", "v2Pro", "v2ProPlus"],
                                scale=5,
                            )

                with gr.Accordion(label=("Pretrained Model Path"), open=False):
                    with gr.Row():
                        with gr.Row(equal_height=True):
                            pretrained_s1 = gr.Textbox(
                                label=("Pretrained GPT Model Path"),
                                value=pretrained_gpt_name[version],
                                interactive=True,
                                lines=1,
                                max_lines=1,
                                scale=3,
                            )
                            pretrained_s2G = gr.Textbox(
                                label=("Pretrained SoVITS-G Model Path"),
                                value=pretrained_sovits_name[version],
                                interactive=True,
                                lines=1,
                                max_lines=1,
                                scale=5,
                            )
                            pretrained_s2D = gr.Textbox(
                                label=("Pretrained SoVITS-D Model Path"),
                                value=pretrained_sovits_name[version].replace("s2G", "s2D"),
                                interactive=True,
                                lines=1,
                                max_lines=1,
                                scale=5,
                            )

                with gr.TabItem(("Dataset Formatting")):
                    with gr.Accordion(label=("output folder (logs/{experiment name}) should have files and folders starts with 23456")):
                        with gr.Row():
                            with gr.Row():
                                inp_text = gr.Textbox(
                                    label=("Text labelling file"),
                                    value=r"D:\RVC1006\GPT-SoVITS\raw\xxx.list",
                                    interactive=True,
                                    scale=10,
                                )
                            with gr.Row():
                                inp_wav_dir = gr.Textbox(
                                    label=("Audio dataset folder"),
                                    # value=r"D:\RVC1006\GPT-SoVITS\raw\xxx",
                                    interactive=True,
                                    placeholder=(
                                        "Please fill in the segmented audio files' directory! The full path of the audio file = the directory concatenated with the filename corresponding to the waveform in the list file (not the full path). If left blank, the absolute full path in the .list file will be used."
                                    ),
                                    scale=10,
                                )

                    with gr.Accordion(process_name_1a):
                        with gr.Row():
                            with gr.Row():
                                gpu_numbers1a = gr.Textbox(
                                    label=("GPU number is separated by -, each GPU will run one process"),
                                    value="%s-%s" % (gpus, gpus),
                                    interactive=True,
                                )
                            with gr.Row():
                                bert_pretrained_dir = gr.Textbox(
                                    label=("Pretrained Chinese BERT Model Path"),
                                    value="GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
                                    interactive=False,
                                    lines=2,
                                )
                            with gr.Row():
                                button1a_open = gr.Button(
                                    value=process_info(process_name_1a, "open"), variant="primary", visible=True
                                )
                                button1a_close = gr.Button(
                                    value=process_info(process_name_1a, "close"), variant="primary", visible=False
                                )
                            with gr.Row():
                                info1a = gr.Textbox(label=process_info(process_name_1a, "info"))

                    with gr.Accordion(process_name_1b):
                        with gr.Row():
                            with gr.Row():
                                gpu_numbers1Ba = gr.Textbox(
                                    label=("GPU number is separated by -, each GPU will run one process"),
                                    value="%s-%s" % (gpus, gpus),
                                    interactive=True,
                                )
                            with gr.Row():
                                cnhubert_base_dir = gr.Textbox(
                                    label=("Pretrained SSL Model Path"),
                                    value="GPT_SoVITS/pretrained_models/chinese-hubert-base",
                                    interactive=False,
                                    lines=2,
                                )
                            with gr.Row():
                                button1b_open = gr.Button(
                                    value=process_info(process_name_1b, "open"), variant="primary", visible=True
                                )
                                button1b_close = gr.Button(
                                    value=process_info(process_name_1b, "close"), variant="primary", visible=False
                                )
                            with gr.Row():
                                info1b = gr.Textbox(label=process_info(process_name_1b, "info"))

                    with gr.Accordion(process_name_1c):
                        with gr.Row():
                            with gr.Row():
                                gpu_numbers1c = gr.Textbox(
                                    label=("GPU number is separated by -, each GPU will run one process"),
                                    value="%s-%s" % (gpus, gpus),
                                    interactive=True,
                                )
                            with gr.Row():
                                pretrained_s2G_ = gr.Textbox(
                                    label=("Pretrained SoVITS-G Model Path"),
                                    value=pretrained_sovits_name[version],
                                    interactive=False,
                                    lines=2,
                                )
                            with gr.Row():
                                button1c_open = gr.Button(
                                    value=process_info(process_name_1c, "open"), variant="primary", visible=True
                                )
                                button1c_close = gr.Button(
                                    value=process_info(process_name_1c, "close"), variant="primary", visible=False
                                )
                            with gr.Row():
                                info1c = gr.Textbox(label=process_info(process_name_1c, "info"))

                    with gr.Accordion(process_name_1abc):
                        with gr.Row():
                            with gr.Row():
                                button1abc_open = gr.Button(
                                    value=process_info(process_name_1abc, "open"), variant="primary", visible=True
                                )
                                button1abc_close = gr.Button(
                                    value=process_info(process_name_1abc, "close"), variant="primary", visible=False
                                )
                            with gr.Row():
                                info1abc = gr.Textbox(label=process_info(process_name_1abc, "info"))

                with gr.TabItem(("Fine-Tuning")):
                    with gr.Accordion(("SoVITS Training: Model Weights saved in SoVITS_weights")):
                        with gr.Row():
                            with gr.Column():
                                with gr.Row():
                                    batch_size = gr.Slider(
                                        minimum=1,
                                        maximum=default_max_batch_size,
                                        step=1,
                                        label=("Batch size"),
                                        value=default_batch_size,
                                        interactive=True,
                                    )
                                    total_epoch = gr.Slider(
                                        minimum=1,
                                        maximum=max_sovits_epoch,
                                        step=1,
                                        label=("Total epochs"),
                                        value=default_sovits_epoch,
                                        interactive=True,
                                    )
                                with gr.Row():
                                    text_low_lr_rate = gr.Slider(
                                        minimum=0.2,
                                        maximum=0.6,
                                        step=0.05,
                                        label=("Text model learning rate weighting"),
                                        value=0.4,
                                        visible=True if version not in v3v4set else False,
                                    )  # v3v4 not need
                                    lora_rank = gr.Radio(
                                        label=("LoRA Rank"),
                                        value="32",
                                        choices=["16", "32", "64", "128"],
                                        visible=True if version in v3v4set else False,
                                    )  # v1v2 not need
                                    save_every_epoch = gr.Slider(
                                        minimum=1,
                                        maximum=max_sovits_save_every_epoch,
                                        step=1,
                                        label=("Save frequency (save_every_epoch):"),
                                        value=default_sovits_save_every_epoch,
                                        interactive=True,
                                    )
                            with gr.Column():
                                with gr.Column():
                                    if_save_latest = gr.Checkbox(
                                        label=("Save only the latest weights"),
                                        value=True,
                                        interactive=True,
                                        show_label=True,
                                    )
                                    if_save_every_weights = gr.Checkbox(
                                        label=("Save a small final model to the 'weights' folder at each save point:"),
                                        value=True,
                                        interactive=True,
                                        show_label=True,
                                    )
    #              -------------------------------Remove----------------------------------------------
                                    if_grad_ckpt = gr.Checkbox(
                                        label="v3是否开启梯度检查点节省显存占用",
                                        value=False,
                                        interactive=True if version in v3v4set else False,
                                        show_label=True,
                                        visible=False,
                                    )  # 只有V3s2可以用
    #              -------------------------------Remove----------------------------------------------
                                with gr.Row():
                                    gpu_numbers1Ba = gr.Textbox(
                                        label=("GPU number is separated by -, each GPU will run one process"),
                                        value="%s" % (gpus),
                                        interactive=True,
                                    )
                        with gr.Row():
                            with gr.Row():
                                button1Ba_open = gr.Button(
                                    value=process_info(process_name_sovits, "open"), variant="primary", visible=True
                                )
                                button1Ba_close = gr.Button(
                                    value=process_info(process_name_sovits, "close"), variant="primary", visible=False
                                )
                            with gr.Row():
                                info1Ba = gr.Textbox(label=process_info(process_name_sovits, "info"))
                    with gr.Accordion(("GPT Training: Model Weights saved in GPT_weights/")):
                        with gr.Row():
                            with gr.Column():
                                with gr.Row():
                                    batch_size1Bb = gr.Slider(
                                        minimum=1,
                                        maximum=40,
                                        step=1,
                                        label=("Batch size per GPU:"),
                                        value=default_batch_size_s1,
                                        interactive=True,
                                    )
                                    total_epoch1Bb = gr.Slider(
                                        minimum=2,
                                        maximum=50,
                                        step=1,
                                        label=("Total training epochs (total_epoch):"),
                                        value=15,
                                        interactive=True,
                                    )
                                with gr.Row():
                                    save_every_epoch1Bb = gr.Slider(
                                        minimum=1,
                                        maximum=50,
                                        step=1,
                                        label=("Save frequency (save_every_epoch):"),
                                        value=5,
                                        interactive=True,
                                    )
                                    if_dpo = gr.Checkbox(
                                        label=("Enable DPO Training"),
                                        value=False,
                                        interactive=True,
                                        show_label=True,
                                    )
                            with gr.Column():
                                with gr.Column():
                                    if_save_latest1Bb = gr.Checkbox(
                                        label=("Save only the latest weights"),
                                        value=True,
                                        interactive=True,
                                        show_label=True,
                                    )
                                    if_save_every_weights1Bb = gr.Checkbox(
                                        label=("Save a small final model to the 'weights' folder at each save point:"),
                                        value=True,
                                        interactive=True,
                                        show_label=True,
                                    )
                                with gr.Row():
                                    gpu_numbers1Bb = gr.Textbox(
                                        label=("GPU number is separated by -, each GPU will run one process"),
                                        value="%s" % (gpus),
                                        interactive=True,
                                    )
                        with gr.Row():
                            with gr.Row():
                                button1Bb_open = gr.Button(
                                    value=process_info(process_name_gpt, "open"), variant="primary", visible=True
                                )
                                button1Bb_close = gr.Button(
                                    value=process_info(process_name_gpt, "close"), variant="primary", visible=False
                                )
                            with gr.Row():
                                info1Bb = gr.Textbox(label=process_info(process_name_gpt, "info"))

            with gr.TabItem(("Inference")):
                gr.Markdown(
                    value=(
                        "Select the model from SoVITS_weights and GPT_weights. The default models are pretrained models for experiencing 5-second Zero-Shot TTS without training."
                    )
                )
                with gr.Row():
                    with gr.Column(scale=2):
                        with gr.Row():
                            GPT_dropdown = gr.Dropdown(
                                label=("GPT weight list"),
                                choices=GPT_names,
                                value=GPT_names[-1] if GPT_names else None,
                                interactive=True,
                            )
                            SoVITS_dropdown = gr.Dropdown(
                                label=("SoVITS weight list"),
                                choices=SoVITS_names,
                                value=SoVITS_names[0] if SoVITS_names else None,
                                interactive=True,
                            )
                    with gr.Column(scale=2):
                        with gr.Row():
                            gpu_number_1C = gr.Textbox(
                                label=("GPU number, can only input ONE integer"), value=gpus, interactive=True
                            )
                            refresh_button = gr.Button(("refresh model paths"), variant="primary")
                with gr.Row(equal_height=True):
                    with gr.Row():
                        batched_infer_enabled = gr.Checkbox(
                            label=("Enable Parallel Inference Version"), value=False, interactive=True, show_label=True
                        )
                        open_tts = gr.Button(
                            value=process_info(process_name_tts, "open"), variant="primary", visible=True
                        )
                        close_tts = gr.Button(
                            value=process_info(process_name_tts, "close"), variant="primary", visible=False
                        )
                    with gr.Column():
                        tts_info = gr.Textbox(label=process_info(process_name_tts, "info"), scale=2)

            pretrained_s2G.change(sync, [pretrained_s2G], [pretrained_s2G_])
            
            button1a_open.click(
                open1a,
                [inp_text, inp_wav_dir, exp_name, gpu_numbers1a, bert_pretrained_dir],
                [info1a, button1a_open, button1a_close],
            )
            button1a_close.click(close1a, [], [info1a, button1a_open, button1a_close])

            button1b_open.click(
                open1b,
                [version_checkbox, inp_text, inp_wav_dir, exp_name, gpu_numbers1Ba, cnhubert_base_dir],
                [info1b, button1b_open, button1b_close],
            )
            button1b_close.click(close1b, [], [info1b, button1b_open, button1b_close])

            button1c_open.click(
                open1c,
                [version_checkbox, inp_text, inp_wav_dir, exp_name, gpu_numbers1c, pretrained_s2G],
                [info1c, button1c_open, button1c_close],
            )
            button1c_close.click(close1c, [], [info1c, button1c_open, button1c_close])

            button1abc_open.click(
                open1abc,
                [
                    version_checkbox,
                    inp_text,
                    inp_wav_dir,
                    exp_name,
                    gpu_numbers1a,
                    gpu_numbers1Ba,
                    gpu_numbers1c,
                    bert_pretrained_dir,
                    cnhubert_base_dir,
                    pretrained_s2G,
                ],
                [info1abc, button1abc_open, button1abc_close],
            )
            button1abc_close.click(close1abc, [], [info1abc, button1abc_open, button1abc_close])

            button1Ba_open.click(
                open1Ba,
                [
                    version_checkbox,
                    batch_size,
                    total_epoch,
                    exp_name,
                    text_low_lr_rate,
                    if_save_latest,
                    if_save_every_weights,
                    save_every_epoch,
                    gpu_numbers1Ba,
                    pretrained_s2G,
                    pretrained_s2D,
                    if_grad_ckpt,
                    lora_rank,
                ],
                [info1Ba, button1Ba_open, button1Ba_close, SoVITS_dropdown, GPT_dropdown],
            )
            button1Ba_close.click(close1Ba, [], [info1Ba, button1Ba_open, button1Ba_close])

            button1Bb_open.click(
                open1Bb,
                [
                    batch_size1Bb,
                    total_epoch1Bb,
                    exp_name,
                    if_dpo,
                    if_save_latest1Bb,
                    if_save_every_weights1Bb,
                    save_every_epoch1Bb,
                    gpu_numbers1Bb,
                    pretrained_s1,
                ],
                [info1Bb, button1Bb_open, button1Bb_close, SoVITS_dropdown, GPT_dropdown],
            )
            button1Bb_close.click(close1Bb, [], [info1Bb, button1Bb_open, button1Bb_close])

            version_checkbox.change(
                switch_version,
                [version_checkbox],
                [
                    pretrained_s2G,
                    pretrained_s2D,
                    pretrained_s1,
                    GPT_dropdown,
                    SoVITS_dropdown,
                    batch_size,
                    total_epoch,
                    save_every_epoch,
                    text_low_lr_rate,
                    if_grad_ckpt,
                    batched_infer_enabled,
                    lora_rank,
                ],
            )
            
            # Dataset Preprocessing Tab Event Handlers
            open_asr_button.click(
                open_asr,
                [asr_inp_dir, asr_opt_dir, asr_model, asr_size, asr_lang, asr_precision],
                [asr_info, open_asr_button, close_asr_button, path_list, inp_text, inp_wav_dir],
            )
            close_asr_button.click(close_asr, [], [asr_info, open_asr_button, close_asr_button])

            open_slicer_button.click(
                open_slice,
                [
                    slice_inp_path,
                    slice_opt_root,
                    threshold,
                    min_length,
                    min_interval,
                    hop_size,
                    max_sil_kept,
                    _max,
                    alpha,
                    n_process,
                ],
                [slicer_info, open_slicer_button, close_slicer_button, asr_inp_dir, denoise_input_dir, inp_wav_dir],
            )
            close_slicer_button.click(close_slice, [], [slicer_info, open_slicer_button, close_slicer_button])

            open_denoise_button.click(
                open_denoise,
                [denoise_input_dir, denoise_output_dir],
                [denoise_info, open_denoise_button, close_denoise_button, asr_inp_dir, inp_wav_dir],
            )
            close_denoise_button.click(close_denoise, [], [denoise_info, open_denoise_button, close_denoise_button])
            
            # Inference Tab Event Handlers
            refresh_button.click(fn=change_choices, inputs=[], outputs=[SoVITS_dropdown, GPT_dropdown])
            
            open_tts.click(
                change_tts_inference,
                [
                    bert_pretrained_dir,
                    cnhubert_base_dir,
                    gpu_number_1C,
                    GPT_dropdown,
                    SoVITS_dropdown,
                    batched_infer_enabled,
                ],
                [tts_info, open_tts, close_tts],
            )
            close_tts.click(
                change_tts_inference,
                [
                    bert_pretrained_dir,
                    cnhubert_base_dir,
                    gpu_number_1C,
                    GPT_dropdown,
                    SoVITS_dropdown,
                    batched_infer_enabled,
                ],
                [tts_info, open_tts, close_tts],
            )
        
        return app