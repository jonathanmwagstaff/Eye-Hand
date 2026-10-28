# Linux system cursor control

## What works on X11, and what changes under Wayland?

### Takeaway
On a real Xorg session, XTEST is the smallest proven path for moving and clicking the global pointer. It is not a portable Wayland path: under Wayland the compositor owns routing and coordinate transforms, so the supported cross-desktop route is a user-authorized RemoteDesktop portal session feeding libei/EIS.

### Cited Findings
- XTEST's `XTestFakeInput` can synthesize `MotionNotify`, `ButtonPress`, and `ButtonRelease` as core input events. — [X.Org XTEST protocol](https://www.x.org/docs/Xext/xtest.pdf)
- `xdotool`/`libxdo` implement mouse and keyboard synthesis through X11 XTEST, and their own documentation warns that Wayland operation is not correct. — [xdotool upstream](https://github.com/jordansissel/xdotool)
- In Wayland, the compositor receives kernel input, chooses the destination surface from its scene graph, and performs coordinate transformations before delivering events; ordinary clients do not own this routing. — [Wayland architecture](https://wayland.freedesktop.org/architecture.html)
- libei is specifically aimed at emulated input in the Wayland stack; the server is typically the compositor, which retains control over whether and when submitted events are processed. — [libei protocol overview](https://libinput.pages.freedesktop.org/libei/)

### Inferences
- The first native proof on this machine can be XTEST-only because the local environment probe reported `XDG_SESSION_TYPE=x11`, `XDG_SESSION_DESKTOP=ubuntu-xorg`, and GNOME. This is an observed development-host fact, not a portability claim.
- Keep the tracker output backend-neutral as normalized `(x, y)` plus explicit button transitions, but implement only one native backend at a time: XTEST for the immediate Xorg proof, then portal/libei before claiming Wayland support.
- Do not use XTEST through XWayland as the Wayland solution: it can affect X clients imperfectly but cannot provide compositor-wide native Wayland control.

### Gaps
- The host probe found no `xdotool`, `ydotool`, `wdotool`, `libei-1.0`, or `liboeffis-1.0` in the current shell; distro packages and development headers must be rechecked on the actual machine before implementation.

## How do the available injection interfaces compare?

### Takeaway
Use RemoteDesktop portal + libei for the real Wayland implementation, XTEST only for the Xorg milestone, and avoid uinput for the product path. Compositor-private protocols and third-party automation CLIs are useful diagnostics or fallbacks, not the first architecture.

### Cited Findings
- The RemoteDesktop portal lifecycle is `CreateSession` → `SelectDevices` → `Start`; `Start` presents user consent and returns the granted devices. It supports pointer-only access. — [XDG RemoteDesktop portal](https://flatpak.github.io/xdg-desktop-portal/docs/doc-org.freedesktop.portal.RemoteDesktop.html)
- The portal recommends `ConnectToEIS()` and libei; after connecting, input is sent over EI and the older D-Bus `Notify*` event methods cannot be mixed into that session. — [XDG RemoteDesktop portal](https://flatpak.github.io/xdg-desktop-portal/docs/doc-org.freedesktop.portal.RemoteDesktop.html)
- `liboeffis` is the official small helper that connects to the session bus, opens the RemoteDesktop session, selects devices, calls `ConnectToEIS()`, and returns the file descriptor to libei. — [libei C libraries](https://libinput.pages.freedesktop.org/libei/libraries/index.html)
- The libei sender API directly supports relative motion, absolute motion, button events, and event frames. — [libei sender API](https://libinput.pages.freedesktop.org/libei/api/group__libei-sender.html)
- The portal also exposes D-Bus `NotifyPointerMotion`, `NotifyPointerMotionAbsolute`, and button methods, so a minimal compatibility implementation can work without libei, although EIS is the documented preferred transport. — [XDG RemoteDesktop portal](https://flatpak.github.io/xdg-desktop-portal/docs/doc-org.freedesktop.portal.RemoteDesktop.html)
- uinput creates a kernel virtual input device by writing to `/dev/uinput`; kernel documentation recommends libevdev rather than raw uinput for new software because it is less error-prone. — [Linux kernel uinput documentation](https://docs.kernel.org/input/uinput.html)
- `ydotool` uses uinput via a persistent daemon and says that daemon usually requires root access to `/dev/uinput`. — [ydotool upstream](https://github.com/ReimuNotMoe/ydotool)
- Direct Wayland compositor-specific approaches are fragmented: for example, the third-party `wdotool` project uses portal/libei for GNOME/KDE but separate `wlr-*` virtual input protocols for wlroots compositors, and labels some desktop-specific paths experimental. — [wdotool upstream](https://github.com/cushycush/wdotool)

### Inferences
- Preferred implementation order: (1) XTEST proof on current Xorg; (2) portal D-Bus `Notify*` proof if minimizing initial dependencies is critical; (3) replace the event transport with `liboeffis` + libei for the maintained Wayland backend. If the project is already C/C++, step 2 may be skipped because the official C helper and libei are direct dependencies anyway.
- Do not add a privileged daemon or broad `/dev/uinput` permission for an assistive desktop app when the compositor-mediated portal exists. uinput bypasses the desktop consent boundary, expands the security impact, and makes multi-monitor mapping the application's problem.
- Do not shell out once per tracking frame to an automation CLI. A persistent XTEST/libei connection avoids process-spawn jitter; CLIs are suitable only for one-shot feasibility checks.

### Gaps
- Portal/backend availability varies by installed desktop and version. The application must query `org.freedesktop.portal.RemoteDesktop` and its version at runtime and report a clear unsupported error rather than assuming GNOME/KDE support.
- No official first-party Python binding for libei was found in the reviewed documentation. The repo's language/build details should determine whether to use the C API directly, a tiny native helper, or the D-Bus `Notify*` path first.

## What permissions and packaging implications apply?

### Takeaway
The portal path deliberately requires desktop-mediated user consent and can persist that consent with a revocable restore token. XTEST needs access to the user's X server; uinput needs write access to a privileged kernel device and should not be granted broadly.

### Cited Findings
- RemoteDesktop `Start()` presents a user dialog, and `SelectDevices()` supports `persist_mode` plus a one-use `restore_token`; persistence can last for the app lifetime or until explicitly revoked, and each successful restore yields a replacement token. — [RemoteDesktop interface definition](https://github.com/flatpak/xdg-desktop-portal/blob/main/data/org.freedesktop.portal.RemoteDesktop.xml)
- Portals are session-bus APIs whose normal model involves the user granting or rejecting access through desktop-specific UI. — [Flatpak portal architecture](https://github.com/flatpak/flatpak/wiki/Portals)
- A portal session can be explicitly closed, and if its D-Bus client vanishes, all of that client's sessions are closed. — [XDG Session interface](https://flatpak.github.io/xdg-desktop-portal/docs/doc-org.freedesktop.portal.Session.html)
- The uinput interface exposes a kernel virtual device through `/dev/uinput`; access to that device is therefore the security boundary. — [Linux kernel uinput documentation](https://docs.kernel.org/input/uinput.html)

### Inferences
- Package the Wayland backend with runtime dependencies on the session portal and a desktop backend; keep saved restore tokens mode `0600`, treat them as revocable credentials, and retry through the consent flow if restoration fails.
- Request POINTER only. Do not request keyboard or touchscreen capabilities until a concrete feature requires them; narrower portal permission is easier for a user to understand and limits mistakes.
- XTEST support should be explicitly labeled Xorg-only and fail closed when the session is Wayland rather than silently targeting XWayland.
- If uinput is ever offered as an advanced fallback, require an administrator-created least-privilege udev/group rule. Never run the tracker itself as root and never recommend world-writable `/dev/uinput`.

### Gaps
- Exact package names and backend versions are distribution-specific and need a target distro matrix before packaging work.
- The portal spec defines persistence, but a particular desktop backend may implement it differently or not at all; this requires live GNOME/KDE tests.

## How should normalized coordinates map across monitor layouts and scaling?

### Takeaway
Do not derive Wayland desktop geometry from physical pixels. Use the logical regions advertised by EIS, select the region/device containing the target, and map normalized tracker output into that region; rebuild the mapping when devices or regions change.

### Cited Findings
- libei regions have logical offsets, widths, and heights controlled by the compositor; they may not match physical desktop pixels (for example, a 1920×1080 logical region can represent a 4K monitor). — [libei region API](https://libinput.pages.freedesktop.org/libei/api/structei__region.html)
- Multi-head layouts may expose distinct absolute devices and nonzero-offset regions, and the client must choose the correct device/region for the desired desktop point. — [libei region API](https://libinput.pages.freedesktop.org/libei/api/structei__region.html)
- An absolute event outside a device's advertised regions is silently discarded, and only one absolute motion request is valid per event frame. — [EI absolute pointer protocol](https://libinput.pages.freedesktop.org/libei/interfaces/ei_pointer_absolute/index.html)
- The portal's D-Bus absolute motion method is defined relative to a ScreenCast stream's logical coordinate space and `logical_size`, requiring a stream identifier. — [XDG RemoteDesktop portal](https://flatpak.github.io/xdg-desktop-portal/docs/doc-org.freedesktop.portal.RemoteDesktop.html)

### Inferences
- For one-monitor v1: clamp normalized inputs to `[0,1]`, then compute `x = offset_x + nx * (width - epsilon)` and the same for `y` in the chosen EIS logical region. This automatically respects compositor scale and rotation because the compositor owns the advertised logical layout.
- For multi-monitor v1, require the user to choose one target monitor during calibration. Full-desktop spanning calibration is a later feature because discontinuous/offset layouts and differently scaled outputs make one naive physical-pixel rectangle incorrect.
- If using the portal's older absolute D-Bus method, pair it with ScreenCast solely to obtain stream/logical geometry. If screen capture is otherwise unnecessary, libei regions are the cleaner final design.
- X11 mapping can use the root screen's X coordinate space, but should be treated as a separate implementation because XRandR scaling/topology rules do not equal EIS region rules.

### Gaps
- The desired user experience for multi-monitor selection versus full-desktop calibration has not been specified.
- Live behavior for monitor hotplug, mixed fractional scaling, and rotated outputs must be verified on the user's compositor even though the EIS model defines the coordinate contract.

## What safety controls and tests are mandatory?

### Takeaway
Ship movement before clicks, require explicit arming, stop on stale tracking data, always release held buttons on every shutdown path, and provide a physical global kill shortcut. Test coordinate and state logic against fakes first, then in a disposable/nested desktop before the real session.

### Cited Findings
- libei requires a sender to call `start_emulating` before events; when stopping, the client is responsible for releasing keys/buttons/touches so they do not remain logically held. — [libei sender API](https://libinput.pages.freedesktop.org/libei/api/group__libei-sender.html)
- EI frames use microsecond `CLOCK_MONOTONIC` timestamps and group changes into one logical hardware event. — [EI device protocol](https://libinput.pages.freedesktop.org/libei/interfaces/ei_device/index.html)
- A paused EI device resets buttons, keys, modifiers, and touches to neutral, and events sent while paused may be discarded. — [EI device protocol](https://libinput.pages.freedesktop.org/libei/interfaces/ei_device/index.html)
- The GlobalShortcuts portal can bind shortcuts that activate regardless of focused application state. — [XDG Global Shortcuts portal](https://flatpak.github.io/xdg-desktop-portal/docs/doc-org.freedesktop.portal.GlobalShortcuts.html)
- Weston can run nested under an existing Wayland compositor or X server, allowing a separate desktop session for manual testing. — [Wayland compositor types](https://wayland.freedesktop.org/docs/book/Compositors.html)
- libei ships demo client/server tools, including an EIS demo server that accepts requests and prints them, which can validate event sequences without controlling the real desktop. — [libei API overview](https://libinput.pages.freedesktop.org/libei/api/index.html)

### Inferences
- Minimum safety state machine: `disarmed → armed-move-only → armed-click`; loss of tracking, stale sample timeout, portal/session close, SIGINT/SIGTERM, exception, or kill shortcut transitions to `disarmed`, emits button-up for every potentially held button, stops emulation, and closes the session.
- A kill shortcut must come from a physical keyboard path and remain available when tracking misbehaves. On Wayland, use GlobalShortcuts rather than trying to detect an injected key in the tracker; initially, a visible disarm control plus Ctrl-C in the launching terminal is acceptable only during supervised development.
- Add hard bounds and a maximum update rate; coalesce tracking samples so each display/event frame gets at most one pointer position. Reject NaN/infinite values and freeze rather than extrapolate when confidence or freshness fails.
- Unit tests need no desktop: feed normalized samples and fake monitor regions, assert clamping/region selection, assert no clicks while disarmed, assert stale input freezes, and assert every exit path emits releases. An integration fake can record XTEST/libei requests.
- Manual rollout: fake sink/log only → browser cursor → nested compositor or separate test X server → real desktop movement only → deliberate dwell/pinch click behind a separate arm action. Keep a physical mouse attached throughout calibration and testing.

### Gaps
- Safe timeout, smoothing, confidence threshold, and maximum pointer speed depend on the actual tracker sample rate and noise; they must be measured from repository/hardware data rather than guessed.
- A nested Weston instance may not expose the same portal backend behavior as GNOME/KDE, so final compositor-specific consent, scaling, hotplug, and emergency-stop tests still require a real but supervised session.
