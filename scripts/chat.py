# UTILS
import os, cv2, subprocess
from PIL      import Image
from stt_vosk import recognize_from_mic

# EXTERNAL FILES
import config
from utils    import build_prompt
from tools    import execute_tool
from message  import MsgBuffer

def main():
    from vllm import LLM, SamplingParams
    from vllm.entrypoints.openai.tool_parsers.hermes_tool_parser import Hermes2ProToolParser
    from vosk import Model

    ## Prepare model parameters
    sampling_params = SamplingParams(**config.SAMPLING_PARAMS)
    system_prompt = build_prompt(config.ROLE_PATH, config.TOOL_PATH)
    conversation  = MsgBuffer(system_prompt, max_groups=5)

    vlm = LLM(
        model=config.VLM_MODEL_PATH,
        generation_config="vllm",
        gpu_memory_utilization=config.VLM_GPU_UTIL,
        max_model_len=config.VLM_MAX_LENGTH,
        limit_mm_per_prompt=config.VLM_LIMIT_INPUT
    )

    parser = Hermes2ProToolParser(tokenizer=vlm.get_tokenizer())

    stt = Model(config.STT_PATH)

    while True:
        print("Write anything to ask:")
        user_input = input().strip()

        config.IMAGE_PATH = os.path.join(config.IMAGE_DIR, "image_1.jpg")
        image = Image.open(config.IMAGE_PATH).convert("RGB")
        conversation.start_user(user_input, image)

        output = vlm.chat(
            messages=conversation.to_messages(),
            sampling_params=sampling_params,
            use_tqdm=True
        )

        answer = output[0].outputs[0].text.strip()
        conversation.append_assistant(answer)
        info = parser.extract_tool_calls(answer, request=None)

        print(conversation)

        # visualize
        image = cv2.imread(config.IMAGE_PATH)
        cv2.imshow("Input Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # if tools are called, execute and get results
        if info.tools_called:
            config.IMAGE_PATH = os.path.join(config.IMAGE_DIR, "rps_rock.jpg")
            image = Image.open(config.IMAGE_PATH).convert("RGB")
            conversation.update_image(image)

            tool_result = execute_tool(info.tool_calls)
            conversation.append_tool(tool_result)

            post_output = vlm.chat(
                messages=conversation.to_messages(),
                sampling_params=sampling_params,
                use_tqdm=True
            )

            post_answer = post_output[0].outputs[0].text.strip()
            conversation.append_assistant(post_answer)

            print(conversation)

        else:
            subprocess.run(["espeak-ng", "-s", "150", "-v", "ko", answer])

if __name__ == "__main__":
    main()