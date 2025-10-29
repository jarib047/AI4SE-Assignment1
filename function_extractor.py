import requests
import os
import ast
import csv
from git import Repo
from tqdm import tqdm
import subprocess
import json

def search_python_repos(min_stars=50, max_repos=1000, token=None):
    headers = {'Authorization': f'token {token}'} if token else {}
    repos = []
    page = 1
    while len(repos) < max_repos:
        url = (
            f'https://api.github.com/search/repositories'
            f'?q=language:Python+stars:>={min_stars}&sort=stars&order=desc&page={page}&per_page=50'
        )
        r = requests.get(url, headers=headers)
        data = r.json()
        repos.extend([repo["clone_url"] for repo in data.get("items", [])])
        if "items" not in data or not data["items"]:
            break
        page += 1
    return repos[:max_repos]


def clone_repos(repo_urls, base_dir="repos"):
    os.makedirs(base_dir, exist_ok=True)
    for url in repo_urls:
        name = url.split("/")[-1].replace(".git", "")
        path = os.path.join(base_dir, name)
        if not os.path.exists(path):
            try:
                print(f"Cloning {name}...")
                Repo.clone_from(url, path, depth=1)
            except Exception as e:
                print(f"Skipping {name}: {e}")


def list_python_files(base_dir="repos"):
    py_files = []
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.endswith(".py") and "test" not in f and not f.startswith("setup"):
                py_files.append(os.path.join(root, f))
    return py_files


def analyze_function(node):
    """Analyze a function AST node and extract 'if' statistics."""
    num_ifs = 0
    total_if_length = 0
    
    for subnode in ast.walk(node):
        if isinstance(subnode, ast.If):
            num_ifs += 1
            if hasattr(subnode, "body") and subnode.body:
                start = subnode.lineno
                end = subnode.body[-1].end_lineno if hasattr(subnode.body[-1], "end_lineno") else start
                total_if_length += (end - start + 1)
    
    return num_ifs, total_if_length


def get_node_end_lineno(node):
    if hasattr(node, "end_lineno") and node.end_lineno:
        return node.end_lineno
    # fallback: get the max lineno of child nodes
    max_line = node.lineno
    for n in ast.walk(node):
        if hasattr(n, "lineno"):
            max_line = max(max_line, n.lineno)
    return max_line


def extract_functions_from_file(file_path):
    """Extract all functions and their properties from a Python file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()
        tree = ast.parse(code)
    except Exception:
        return []

    functions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            try:
                # start, end = node.lineno - 1, node.end_lineno
                # snippet = "\n".join(code.splitlines()[start:end])
                start = node.lineno - 1
                end = get_node_end_lineno(node)
                snippet = "\n".join(code.splitlines()[start:end])
                num_ifs, total_if_len = analyze_function(node)
                functions.append({
                    "function_name": node.name,
                    "function_length": end - start,
                    "num_ifs": num_ifs,
                    "total_if_length": total_if_len,
                    "function_code": snippet
                })
            except Exception:
                continue
    return functions


def get_repo_commit_sha(repo_path):
    """Return the latest commit SHA of a repository."""
    try:
        repo = Repo(repo_path)
        return repo.head.commit.hexsha
    except Exception:
        return None


def find_repo_root(file_path):
    path = os.path.abspath(file_path)
    while path != "/":
        if os.path.isdir(os.path.join(path, ".git")):
            return path
        path = os.path.dirname(path)
    return None


# def build_function_corpus(py_files, output_csv="functions.csv"):
#     """Extract all functions and metadata into a CSV file."""
#     os.makedirs(os.path.dirname(output_csv), exist_ok=True)

#     with open(output_csv, "w", encoding="utf-8", newline="") as csvfile:
#         writer = csv.DictWriter(csvfile, fieldnames=[
#             "repo_name",
#             "commit_sha",
#             "function_name",
#             "function_length",
#             "num_ifs",
#             "total_if_length",
#             "function_code"
#         ])
#         writer.writeheader()

#         for file_path in tqdm(py_files):
#             # repo_root = file_path.split(os.sep)[1]  # assumes repos/<repo_name>/...
#             # repo_path = os.path.join("repos", repo_root)
#             repo_path = find_repo_root(file_path)
#             repo_root = os.path.basename(repo_path)
#             commit_sha = get_repo_commit_sha(repo_path)

#             functions = extract_functions_from_file(file_path)
#             for fn in functions:
#                 writer.writerow({
#                     "repo_name": repo_root,
#                     "commit_sha": commit_sha,
#                     **fn
#                 })
#     subprocess.run(["rm", "-rf", "repos"], check=True)
#     print(f"Saved functions metadata to {output_csv}")


def build_function_corpus_json(py_files, output_json="functions.json"):
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    all_functions = []

    for file_path in tqdm(py_files):
        repo_path = find_repo_root(file_path)  # see previous helper
        if repo_path is None:
            continue
        repo_name = os.path.basename(repo_path)
        commit_sha = get_repo_commit_sha(repo_path)

        functions = extract_functions_from_file(file_path)
        for fn in functions:
            fn_data = {
                "repo_name": repo_name,
                "commit_sha": commit_sha,
                **fn
            }
            all_functions.append(fn_data)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(all_functions, f, indent=2)

    print(f"Saved functions metadata to {output_json}")


if __name__ == "__main__":
    repo_urls = search_python_repos(max_repos=3000)
    clone_repos(repo_urls, base_dir="python_extractor/repos")
    py_files = list_python_files(base_dir="python_extractor/repos")
    # build_function_corpus(py_files, output_csv="python_extractor/datasets/functions.csv")
    build_function_corpus_json(py_files, output_json="python_extractor/datasets/functions.json")