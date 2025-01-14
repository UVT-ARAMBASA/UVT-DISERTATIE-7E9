import pkg_resources

def list_installed_libraries():
    # Get the list of installed packages
    installed_packages = pkg_resources.working_set

    # Format and display the libraries
    print(f"{'Library':<30} {'Version'}")
    print("=" * 40)
    for package in sorted(installed_packages, key=lambda x: x.project_name.lower()):
        print(f"{package.project_name:<30} {package.version}")

if __name__ == "__main__":
    list_installed_libraries()

