# Third-Party Licenses and Attribution

This project uses and builds upon the following third-party components:

## Base Model

**Qwen/Qwen2.5-Coder-0.5B-Instruct**
- Source: https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct
- License: Apache License 2.0
- Copyright: Qwen Team, Alibaba Cloud
- Description: Base language model for code generation

### Apache License 2.0 Summary
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

## MLX Model Weights

**mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit**
- Source: https://huggingface.co/mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit
- License: Apache License 2.0 (inherited from base model)
- Description: MLX-optimized 4-bit quantized version of Qwen2.5-Coder-0.5B-Instruct
- Conversion: Community contribution for Apple Silicon optimization

## Training Dataset

**flwrlabs/code-alpaca-20k**
- Source: https://huggingface.co/datasets/flwrlabs/code-alpaca-20k
- License: Apache License 2.0
- Description: Code instruction dataset based on Stanford Alpaca methodology
- Size: 20,000 code instruction-following examples

## Python Dependencies

### MLX-LM
- License: MIT License
- Description: MLX language model utilities
- Source: https://github.com/ml-explore/mlx-lm

### Hugging Face Datasets
- License: Apache License 2.0
- Description: Dataset loading and processing library
- Source: https://github.com/huggingface/datasets

### Hugging Face Hub
- License: Apache License 2.0
- Description: Hugging Face Hub client library
- Source: https://github.com/huggingface/huggingface_hub

### PyYAML
- License: MIT License
- Description: YAML parser and emitter
- Source: https://github.com/yaml/pyyaml

## Disclaimers

### No Endorsement
This project is not endorsed by, affiliated with, or sponsored by:
- Qwen Team or Alibaba Cloud
- The MLX community
- flwrlabs or the code-alpaca-20k dataset authors
- Hugging Face

### Attribution Requirements
When using this model or its derivatives:
1. Maintain attribution to the base model (Qwen2.5-Coder-0.5B-Instruct)
2. Maintain attribution to the training dataset (code-alpaca-20k)
3. Include this license file or equivalent attribution
4. Do not imply endorsement by original authors

### Modifications
This project provides:
- LoRA adapter weights (fine-tuning on top of base model)
- Training and serving infrastructure
- Documentation and usage examples

This project does NOT redistribute:
- Base model weights (users download from original source)
- Complete fine-tuned model weights
- Training dataset (users download from original source)

## License Compliance

All components used in this project are licensed under permissive open-source licenses (Apache-2.0, MIT) that allow:
- Commercial use
- Modification
- Distribution
- Private use

Users must:
- Include copyright notices
- Include license text
- State changes made
- Not use trademarks without permission

## Full License Texts

### Apache License 2.0
Full text available at: http://www.apache.org/licenses/LICENSE-2.0

### MIT License
Full text available at: https://opensource.org/licenses/MIT

## Questions

For questions about licensing or attribution, please open an issue at:
https://github.com/salakash/AskBuddyX/issues