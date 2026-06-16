This folder includes the source code of the Oculus Quest APK delivered in the repository. Use it if you intend to customize the app. Before following this instruction, please read the [README.md](../README.md) in the repository root folder.

In order to compile the code you need to download the Oculus Quest SDK version 1.50.0 ([link](https://developer.oculus.com/downloads/package/oculus-mobile-sdk/1.50.0/)) and **place this repository in the root folder of the unpacked SDK**. Please keep the repository name as 'oculus_reader' to conform to the predefined project configuration.

## Development

### Preparation

- (Mac only) Install Xcode
- Install Android development studio. On Ubuntu you can run `sudo snap install android-studio --classic`, otherwise follow: <https://developer.android.com/studio>
- Got to Tools -> SDK Manager and install Android 8.0 (API Level 26) platform.
- Select *Open an Existing Project*, find the repository path, and open: app_source/Projects/Android/build.gradle
- Go to File -> Project structure...
- Select Gradle 6.1.1
- You will get an error that NDK is missing. Go to Tools -> SDK Manager and select SDK Tools tab
    - Select "Show Package Details" on the right side below the list to enable version selection
    - Scroll down to "NDK (Side by side)"
    - Select and install version which is close to the default one indicated in the error
- In SDK root create the `local.properties` file and provide following content
    - on Mac:
    ```
    sdk.dir=/Users/<username>/Library/Android/sdk/
    ndk.dir=/Users/<username>/Library/Android/sdk/ndk/<NDK_version>
    ```
    - on Ubuntu:
    ```
    sdk.dir=/home/<username>/Android/Sdk
    ndk.dir=/home/<username>/Android/Sdk/ndk/<NDK_version>
    ```
- In SDK root edit the `settings.properties` file. Replace the the `rootProject.name` and `include` lines with:
```
rootProject.name = "OculusTeleop"

include ':VrSamples:SampleFramework:Projects:Android'
include ':oculus_reader:app_source:Projects:Android'
```
- (Windows only) Install drivers https://developer.oculus.com/downloads/package/oculus-adb-drivers/ 

### Development

- Select *Open an Existing Project*, find the repository path, and open: app_source/Projects/Android/build.gradle
- The source code resides in the 'app_source/Src' folder.
- When ready, build the APK with menu Build -> Build App Bundle(s) / APK(s) -> Build APK(s). Locate file and replace the APK in the repository
- Reinstall APK with `python oculus_reader/install.py --reinstall`

## Run the code

Follow the instruction in [oculus_reader/README.md](../oculus_reader/README.md) to install the required packages, configure path to new the APK (as described in the same README file), and then finish up with the [README.md](../README.md) in the root folder to run the code.
