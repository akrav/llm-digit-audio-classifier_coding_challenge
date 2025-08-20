import os

def test_initial_structure_exists():
    required_dirs = [
        "src",
        "data",
        "models",
        "tests",
        "Build Documentation",
        "Tickets",
    ]
    for d in required_dirs:
        assert os.path.isdir(d), f"Missing directory: {d}"

    required_files = [
        "README.md",
        "requirements.txt",
        "Build Documentation/structure.md",
        "Build Documentation/Sprint-Progress.md",
        "Build Documentation/Troubleshooting.md",
    ]
    for p in required_files:
        assert os.path.isfile(p), f"Missing file: {p}"
