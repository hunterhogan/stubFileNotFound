\
import argparse
import os
import pathlib
import subprocess
import sys

# Define the root directory of the project and the stubs directory
ROOT_DIR = pathlib.Path(__file__).parent.parent.resolve()
STUBS_DIR = ROOT_DIR / "stubs"

def run_command(command: list[str], cwd: str | None = None) -> None:
    """Runs a command and prints its output."""
    print(f"Running command: {' '.join(command)}")
    try:
        process = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            cwd=cwd,
            encoding='utf-8',
            errors='ignore' # Ignore encoding errors during capture
        )
        print("Command output:")
        print(process.stdout)
        if process.stderr:
            print("Command error output:")
            print(process.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        print("Error output:")
        print(e.stderr)
        sys.exit(e.returncode)
    except FileNotFoundError:
        print(f"Error: Command not found: {command[0]}. Is it installed and in PATH?")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)


def generate_stubs(package_name: str, use_stubgen_pyx: bool) -> None:
    """Generates stubs for a given package."""
    output_dir = STUBS_DIR / package_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Ensured output directory exists: {output_dir}")

    if use_stubgen_pyx:
        print(f"Attempting to generate stubs for Cython package '{package_name}' using stubgen-pyx...")
        # stubgen-pyx needs the path to the installed package
        try:
            import importlib.util
            spec = importlib.util.find_spec(package_name)
            if spec and spec.origin:
                package_path = pathlib.Path(spec.origin).parent
                print(f"Found package path: {package_path}")
                cmd = [sys.executable, "-m", "stubgen_pyx", str(package_path), "--output-dir", str(output_dir)]
                run_command(cmd)
            else:
                print(f"Error: Could not find installation path for package '{package_name}'.")
                sys.exit(1)
        except ImportError:
            print("Error: stubgen-pyx does not seem to be installed or importable.")
            print("Please install it: pip install stubgen-pyx")
            sys.exit(1)
        except Exception as e:
             print(f"Error finding package path for {package_name}: {e}")
             sys.exit(1)

    else:
        print(f"Generating stubs for Python package '{package_name}' using stubgen...")
        # stubgen uses the package name directly
        cmd = [
            sys.executable,
            "-m",
            "mypy.stubgen",
            "--output", str(STUBS_DIR),
            "--package", package_name,
            "--include-private",
            "--include-docstrings",
            "--verbose",
        ]
        run_command(cmd)

    # Modernize the generated stubs
    print(f"Modernizing generated stubs in {output_dir}...")
    pyi_files = list(output_dir.rglob("*.pyi"))
    if pyi_files:
        pyupgrade_cmd = [
            sys.executable,
            "-m",
            "pyupgrade",
            "--py310-plus",
        ] + [str(f) for f in pyi_files]
        run_command(pyupgrade_cmd)
    else:
        print("No .pyi files found to modernize.")

    print(f"Successfully generated and modernized stubs for {package_name} in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and modernize .pyi stubs for Python packages.")
    parser.add_argument("package_name", help="The name of the installed Python package to generate stubs for.")
    parser.add_argument("--cython", action="store_true", help="Use stubgen-pyx for Cython packages.")

    args = parser.parse_args()

    generate_stubs(args.package_name, args.cython)
