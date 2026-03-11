import os

def apply_inference_patch():
    inference_path = os.path.join("E:\\tts", "fish-speech", "fish_speech", "models", "text2semantic", "inference.py")
    if not os.path.exists(inference_path):
        return False
        
    with open(inference_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    # Check if already patched
    if "# [PATCHED] Safely extract generated codes" in content:
        return True
        
    # Replace the faulty block in generate_long
    target_block = """            # Extract generated codes
            codes = y[1:, prompt_length:-1].clone()
            assert (codes >= 0).all(), f"Negative code found: {codes}"

            # Add assistant message with generated codes back to conversation
            conversation.append(
                Message(
                    role="assistant",
                    parts=[VQPart(codes=codes.cpu(), cal_loss=False)],
                    cal_loss=False,
                    modality="voice",
                    add_im_start=True,
                    add_im_end=True,
                )
            )

            yield GenerateResponse(action="sample", codes=codes, text=batch_text)"""
            
    replacement_block = """            # [PATCHED] Safely extract generated codes
            if y.size(1) > prompt_length + 1:
                codes = y[1:, prompt_length:-1].clone()
                assert (codes >= 0).all(), f"Negative code found: {codes}"
                
                # Add assistant message with generated codes back to conversation
                conversation.append(
                    Message(
                        role="assistant",
                        parts=[VQPart(codes=codes.cpu(), cal_loss=False)],
                        cal_loss=False,
                        modality="voice",
                        add_im_start=True,
                        add_im_end=True,
                    )
                )

                yield GenerateResponse(action="sample", codes=codes, text=batch_text)
            else:
                from loguru import logger
                logger.warning(f"Batch {batch_idx}: No valid semantic tokens generated beyond prompt.")
                # Yield empty tensor to prevent upstream slice exception
                empty_codes = torch.empty((model.config.num_codebooks, 0), dtype=y.dtype, device=y.device)
                yield GenerateResponse(action="sample", codes=empty_codes, text=batch_text)"""
                
    content = content.replace(target_block, replacement_block)
    
    # Adding the second fix for `decode_n_tokens` empty lists
    target_block2 = """    del cur_token

    return torch.cat(new_tokens, dim=1)"""
    
    replacement_block2 = """    del cur_token

    # [PATCHED] Prevent empty lists from crashing torch.cat
    if not new_tokens:
        import logging
        logging.getLogger(__name__).warning("No new tokens were generated. Returning empty tensor.")
        return torch.empty((model.config.num_codebooks + 1, 0), dtype=torch.int, device=model.device)

    return torch.cat(new_tokens, dim=1)"""
    
    content = content.replace(target_block2, replacement_block2)
    
    with open(inference_path, "w", encoding="utf-8") as f:
        f.write(content)
        
    print("Successfully patched `fish_speech/models/text2semantic/inference.py` bounds!")
    return True

if __name__ == "__main__":
    apply_inference_patch()
