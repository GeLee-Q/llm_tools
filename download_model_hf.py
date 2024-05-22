from huggingface_hub import snapshot_download

repo_id = "internlm/internlm-chat-7b"  # 模型在Hugging Face上的名称
local_dir = "/workspace/internlm-chat-7b"  # 本地模型存储的地址
local_dir_use_symlinks = False  # 本地模型使用文件保存，而非blob形式保存
token = "xxxxxxxxxxxxxxxxxxxxxxxxxxx"  # 在Hugging Face上生成的 access token

# 下载模型
snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    local_dir_use_symlinks=local_dir_use_symlinks,
    token=token
)