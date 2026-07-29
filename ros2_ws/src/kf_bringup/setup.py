from setuptools import setup

package_name = "kf_bringup"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        # Ament package marker.
        ("share/ament_index/resource_index/packages", ["resource/kf_bringup"]),
        (f"share/{package_name}", ["package.xml"]),
        # Launch and config files installed into share/ — listed by explicit filename, NOT a
        # glob: a new launch file or yaml is silently not installed unless it is added here.
        (
            f"share/{package_name}/launch",
            ["launch/eskf_kitti.launch.py", "launch/tracker_synthetic.launch.py",
             "launch/full_pipeline.launch.py"],
        ),
        (
            f"share/{package_name}/config",
            ["config/eskf_kitti.yaml", "config/tracker.yaml", "config/full_pipeline.yaml"],
        ),
    ],
    install_requires=["setuptools"],
    # This is what makes `colcon test` run the pytest suite in test/. colcon's Python test task
    # picks its testing step by calling has_test_dependency(setup_py_data, "pytest") — with no
    # tests_require it falls through to the deprecated `setup.py test` step, which finds no
    # unittest suite and reports "Ran 0 tests" while exiting 0. A silent zero, not a failure.
    tests_require=["pytest"],
    zip_safe=True,
    maintainer="Derek Wang",
    maintainer_email="info@prandtldynamics.com",
    description="Launch files, config, and replay nodes for the kf_eskf and kf_tracker stacks.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "kitti_replay = kf_bringup.kitti_replay:main",
            "tracking_replay = kf_bringup.tracking_replay:main",
            "pipeline_replay = kf_bringup.pipeline_replay:main",
            "viz_markers = kf_bringup.viz_markers:main",
            "lidar_replay = kf_bringup.lidar_replay:main",
            "image_replay = kf_bringup.image_replay:main",
        ],
    },
)
