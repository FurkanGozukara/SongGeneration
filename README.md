# SECourses LeVo Song Generator - This is the Very Best local Song generator APP - I can say almost level of Suno AI Music - We have so many batch feature

## App Download Link : https://www.patreon.com/posts/135592123

## Made for Only SECourses Patreon Subscribers Please Use Above Link to Get Zip File and Installers

### App download and install link : https://www.patreon.com/posts/135592123

## Demo Song and Mini App Tutorial : https://www.youtube.com/watch?v=t_uJ50LZVss

# SECourses LeVo Song Generator

> A powerful local AI song generation app built around Tencent AI Lab's SongGeneration models, with an easy Windows workflow, advanced VRAM controls, preset management, batch generation, and reference voice support.

---

## Overview

**SECourses LeVo Song Generator** is a local Windows app for AI music generation with a strong focus on usability, control, and quality. It wraps the official **SongGeneration** project in a much more practical desktop workflow and adds a large number of quality-of-life improvements for real-world generation.

This app includes:

- Batch processing
- Multiple generation modes
- Full preset save/load support
- 20 pre-made presets
- Generate with each preset automatically
- Better VRAM handling and optimization options
- Reference voice upload support
- Optional advanced quantization settings
- Windows installer/update scripts
- Automatic model download flow

Official upstream repository:  
https://github.com/tencent-ailab/SongGeneration

Latest installer zip:  
**Levo_Song_Gen_v11.zip**

---

## Screenshots and Demo

### Public screenshots gallery
https://www.reddit.com/r/SECourses/comments/1r9g2d8/secourses_levo_song_generator_this_is_the_very/

### Example generations and feature showcase
https://youtu.be/t_uJ50LZVss

### Installation tutorial
If installation fails, make sure you install everything exactly as shown here:  
https://youtu.be/DrhUHnYfwC0

---

## Main Features

### Song generation
- Generate songs locally on Windows
- Supports the latest SongGeneration models integrated into the app
- Duration control now follows the selected target length much more accurately
- Can generate mixed audio, vocals only, background music only, or separate outputs depending on settings/version

### Presets and workflow
- Full preset saving and loading
- 20 pre-made presets included
- Generate a song with each preset automatically
- Updated default presets for better output quality
- Example lyrics included and updated over time

### Batch tools
- Batch processing support
- Number-of-generations workflow
- Useful for rapid iteration and prompt testing

### Input and prompting
- Reference voice upload support
- Audio input support
- Video input support with automatic audio extraction
- Optional extra description prompt
- Optional “Use only extra prompt” mode for full manual prompt control
- BPM slider support
- Improved description assembly for better transparency in CLI output

### Generation controls
- Generation type selection:
  - Vocals + BGM (mixed)
  - Vocals only
  - BGM only
  - Separate output modes where supported
- Song structure tag support
- Song structure tag checking system updated for newer versions
- Better generation progress display in the command window

### Performance and VRAM optimization
- Auto VRAM optimization
- Advanced quantization options:
  - Enable LLM MLP-Only Int8
  - Enable LLM MLP-Only Int4
- Block swapping option in advanced settings
- Reduced VRAM usage depending on settings and duration
- Better handling for out-of-memory situations

---

## What Makes This App Stand Out

Compared with a raw upstream install, this app is designed to make the SongGeneration model family much easier to use in practice.

Highlights include:

- A much more user-friendly local workflow
- Integrated install/update scripts
- Automatic model download support
- Preset-based experimentation
- Batch generation workflows
- Advanced VRAM management options
- Reference voice fixes and improvements
- Frequent feature updates and bug fixes

---

## Windows Requirements

You will need:

- **Windows**
- **Python 3.10**
- **FFmpeg**
- **CUDA 12.8**
- **C++ Build Tools**
- **MSVC**
- **Git**

If something does not work, follow the installation tutorial exactly:  
https://youtu.be/DrhUHnYfwC0

---

## Installation

## Fresh install recommended

For major updates, a **fresh install into a new folder** is strongly recommended.

### Standard install/update steps

1. Download the latest zip:
   - **Levo_Song_Gen_v11.zip**

2. Either:
   - Extract into a **new folder**, or
   - Overwrite your old files with the latest package

3. Run:
   - `Windows_Install_and_Update.bat`

4. Wait until installation fully finishes

5. The installer will automatically download required models

6. Start the app with:
   - `Windows_Start_App.bat`

---

## Updating

For the latest versions, the recommended update flow is:

1. Download the newest zip
2. Extract to a new folder or overwrite previous files
3. Run `Windows_Install_and_Update.bat`
4. Let model downloads finish
5. Start with `Windows_Start_App.bat`

For some older versions, a **fresh install** was specifically recommended due to model and dependency changes.

---

## Model Notes

- The app now tracks newer SongGeneration model releases over time
- Older models were removed in some updates as defaults changed
- In later releases, the default model became:
  - `songgeneration_v2_large`

The app also includes automatic model download support.

To download or resume models in some versions, helper scripts may be included, such as:

- `Windows_Download_or_Resume_Models.bat`
- `Windows_Resume_Download_Models.bat`

---

## Reference Voice and Input Audio Notes

Reference voice input is limited to **10 seconds**.

Reason:
- The model works like an LLM with a fixed context size
- Input audio consumes a significant portion of the available context window
- The context size is believed to be around **8k**

This means shorter reference audio is necessary to preserve enough room for generation.

---

## VRAM and Performance Notes

This project uses an **autoregressive LLM-style architecture** for song token generation, so VRAM behavior is heavily influenced by **KV-cache usage**.

Important notes:

- Shorter song duration uses less VRAM
- Even with Int8 or Int4 options, peak VRAM may not drop dramatically because KV-cache remains a major factor
- Block swapping can help in out-of-memory situations, but it can be very slow
- Because of the architecture, block swapping cannot fully solve VRAM pressure in every case

### Recommended setting
- **Int8 is recommended**

A reported example from the project notes:
- A **180 second** song can take as low as **12 GB VRAM** with Int8 activation in the updated setup

---

## Usage Tips

### Recommended lyric structure
Based on project notes, simpler structures may work better, such as:

- `[verse]`
- `[chorus]`

Potentially useful tags:
- `[drop]`
- `[hook]`
- `[instrumental]`

Less emphasis was recommended on tags like:
- `[pre-chorus]`
- `[post-chorus]`

### Extra description examples
You can guide style and energy with prompts such as:

- `pop, uplifting, the bpm is 130`
- `male, bright, rock, uplifting, electric guitar and drums, the bpm is 135`

### Duration behavior
In newer versions:
- The model follows the selected **Duration (seconds)** much more accurately
- If your lyrics are too short, you can trim the generated song afterward where the output naturally ends

---

## Cloud / Remote GPU Options

## Massed Compute (recommended cloud option)

Register here:  
https://vm.massedcompute.com/signup?linkId=lp_034338&sourceId=secourses&tenantId=massed-compute

Coupon:
- `SECourses`

Notes:
- Coupon works on all GPUs
- H100 is noted as a strong option for speed/value
- RTX A6000 ADA can also be used

Additional details:  
https://www.patreon.com/posts/26671823

Then:

1. Select **SECourses** from the Creator dropdown
2. Follow `Massed_Compute_Instructions_READ.txt`

Tutorial example:  
https://youtu.be/KW-MHmoNcqo?si=G1WbG-Qw4ujWvOtG&t=778

---

## RunPod

Register here:  
https://get.runpod.io/955rkuppqv4h

Then follow:

- `Runpod_Instructions_READ.txt`
- `Runpod_SimplePod_Levo_Song_Instructions.txt`

Tutorial example:  
https://youtu.be/KW-MHmoNcqo?si=QN8X8Sjn13ZYu-EU&t=1323

---

## Changelog

## 15 March 2026 — V11.0

A major update with large improvements.

### Changes
- Fixed the issue with uploading reference voice
- Updated the **Song Structure Tags** tab
- Updated the automatic tag checking system for the latest version
- Model now obeys the selected **Duration (seconds)** much more accurately
- Lower duration now uses less VRAM
- Added **Auto VRAM Set Optimize**
- Added:
  - **Enable LLM MLP-Only Int8**
  - **Enable LLM MLP-Only Int4**
- Int8 recommended
- Implemented new **block swapping** in advanced settings
- Improved VRAM efficiency
- Example lyrics updated
- Various bugs fixed
- General improvements made

### Notes
- A 180-second song may go as low as 12 GB VRAM with Int8
- Most VRAM is still used by KV-cache due to the autoregressive architecture
- Block swapping can help if you get OOM, but it is slow and cannot fully solve KV-cache-heavy usage

### Update instructions
- Download the latest **v11** zip
- Overwrite old files or extract into a new folder
- Run `Windows_Install_and_Update.bat`
- Wait for full install and automatic model download
- Start with `Windows_Start_App.bat`

---

## 13 March 2026 — V10.1

### Changes
- VRAM management improved further
- New published model `songgeneration_v2_large` integrated
- This became the default model
- Previous models removed
- CMD progress display improved further
- Generation Type accuracy improved:
  - mixed
  - vocal only
  - bgm only
  - separate outputs
- Latest upstream changes implemented
- App now includes all official features and more

### Recommendation
- Fresh install recommended using the latest V10 zip
- Install into a new folder
- Verify setup
- Delete older installation afterward

---

## 20 February 2026 — V9.0

### Changes
- Installers upgraded to latest version
- Switched to **uv** installation
- Installation speed improved dramatically
- Several bugs fixed
- App stability improved

### Recommendation
- Fresh install with latest zip
- Delete the older directory completely afterward

---

## 9 November 2025 — V7.0

### Changes
- App upgraded to newest models and newest libraries
- Fresh install recommended
- Added support for **SongGeneration Large** as the best-quality model tested at the time
- Can generate songs up to **4.5 minutes**
- Automatic download of the large model included

### Additional note
To download other models, use:
- `Windows_Download_or_Resume_Models.bat`

---

## 4 August 2025 — V4

### Changes
- Fixed broken audio input that caused model loading to get stuck
- Added support for both **audio and video upload**
- App now auto-extracts audio from video files
- Added `Windows_Resume_Download_Models.bat` to resume failed downloads
- Added **Extra Description (Optional)** feature
- Added **Use only extra prompt** mode
- Added **BPM slider**
- Improved final description display in CMD
- Moved **Generation Type** to the main tab
- Supports:
  - vocals + BGM (mixed)
  - vocals only
  - BGM only
- Updated default preset for better results
- Continued workflow experimentation and optimization
- Model capable of generating songs up to **2.5 minutes**

### Recommended usage note
Project notes suggested simple structures like `[verse]` and `[chorus]` may work especially well.

---

## Community and Support

### Discord
Join the SECourses Discord for help, discussion, and support.  
You can also share your Discord username to receive your special rank.

### Patreon resources
Patreon includes:
- Exclusive post index to find scripts easily
- Exclusive update index to see which scripts were updated or added last
- Special generative scripts lists for broader workflows

### GitHub / Reddit / LinkedIn
Please support the ecosystem by:
- Starring, watching, and forking the GitHub repository
- Joining the subreddit
- Following on LinkedIn

---

## Troubleshooting

If the app does not work:

1. Confirm all Windows requirements are installed
2. Follow the video tutorial exactly:
   - https://youtu.be/DrhUHnYfwC0
3. Try a fresh install into a new folder
4. Re-run `Windows_Install_and_Update.bat`
5. Allow the model downloads to finish completely
6. Use lower duration if you are running out of VRAM
7. Enable Int8 optimization
8. Try block swapping only if necessary, understanding that it may be slow

The app is described as being very close to complete, but bug reports are still welcome.

---

## Disclaimer

This project is built around the official SongGeneration repository and related models, with significant workflow, install, usability, and optimization improvements added in the SECourses app setup.

Official repository:  
https://github.com/tencent-ailab/SongGeneration

This project is not presented here as an official Tencent release unless explicitly stated by the original authors.

---

## Credits

- **SECourses** — app workflow, tooling, packaging, optimization features, guides, training, scripts, and community support
- **Tencent AI Lab** — official SongGeneration repository and core model work

---

## Quick Start

1. Download the latest zip
2. Extract it into a folder
3. Run `Windows_Install_and_Update.bat`
4. Wait for setup and models to finish downloading
5. Run `Windows_Start_App.bat`
6. Choose your settings, lyrics, duration, and generation mode
7. Generate songs locally

---

## Links

- Official repo: https://github.com/tencent-ailab/SongGeneration
- Latest screenshots gallery: https://www.reddit.com/r/SECourses/comments/1r9g2d8/secourses_levo_song_generator_this_is_the_very/
- Feature demo: https://youtu.be/t_uJ50LZVss
- Installation tutorial: https://youtu.be/DrhUHnYfwC0
- Massed Compute signup: https://vm.massedcompute.com/signup?linkId=lp_034338&sourceId=secourses&tenantId=massed-compute
- RunPod signup: https://get.runpod.io/955rkuppqv4h

---

## Final Note

If you enjoy the app, please consider supporting the project by joining the community, reporting bugs, and following the related GitHub, Discord, Reddit, Patreon, and LinkedIn channels.
