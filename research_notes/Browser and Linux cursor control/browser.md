# Browser cursor prototype

## Best minimal transport from the native tracker to the page

### Takeaway
Use one loopback-only HTTP server that serves both the demo page and a server-sent event (SSE) endpoint. SSE matches the required one-way tracker-to-page flow, works through the browser-native `EventSource` API, reconnects automatically, and can be implemented with Python 3.12's standard library; WebSocket should wait until there is a real need for frequent bidirectional traffic.

### Cited Findings
- SSE is explicitly a one-way server-to-browser connection, uses `text/event-stream`, supports named events, and automatically reconnects after a dropped connection unless the page closes it. — [MDN: Using server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events)
- The HTML standard defines `EventSource`, UTF-8 event framing, event IDs/reconnection, and `204 No Content` as a way for a server to stop reconnection. — [WHATWG HTML: Server-sent events](https://html.spec.whatwg.org/dev/server-sent-events.html)
- WebSocket is bidirectional, but its ordinary browser API has no backpressure; excessive incoming messages can buffer in memory or make the page unresponsive. — [MDN: WebSocket API](https://developer.mozilla.org/en-US/docs/Web/API/WebSockets_API)
- WebRTC data channels require peer-connection negotiation and an out-of-band signaling mechanism, even though the eventual channel is peer-to-peer. — [MDN: Using WebRTC data channels](https://developer.mozilla.org/en-US/docs/Web/API/WebRTC_API/Using_data_channels); [MDN: Signaling](https://developer.mozilla.org/en-US/docs/Web/API/WebRTC_API/Signaling_and_video_calling)
- Python 3.12 includes `HTTPServer`, `ThreadingHTTPServer`, and `BaseHTTPRequestHandler`; the documentation specifically notes that the threaded variant handles browsers that pre-open sockets. The module is not recommended for production because it implements only basic security checks. — [Python 3.12 `http.server`](https://docs.python.org/3.12/library/http.server.html)
- The repository currently targets Python 3.12 and has no HTTP/WebSocket framework in runtime dependencies. — [Repository manifest](../../pyproject.toml)

### Inferences
- Minimal wire format: emit a named `pointer` SSE event whose JSON data is only `{seq, x, y, pressed}`. `x` and `y` are finite normalized values; `seq` lets the page reject duplicates/out-of-order data; a false-to-true `pressed` edge activates once. Keep calibration/model-specific data out of this browser boundary.
- Serve the HTML and `/events` from the same `127.0.0.1` origin. This avoids CORS configuration, adds no dependency, and prevents LAN exposure. Use `ThreadingHTTPServer` only as a local proof-of-concept server, not as a production service.
- Publish the newest tracker sample at no more than the intended display/update rate (initially 60 Hz), rather than queueing every camera inference. A single-slot latest-value handoff prevents reliable TCP delivery from turning old samples into cursor lag.
- Rare page-to-process actions such as “recalibrate” can use a normal same-origin `POST` later. Switch to WebSocket only if frequent bidirectional messages become a demonstrated requirement. WebRTC has no justification on one machine.

### Gaps
- SSE latency at 30–60 updates/second through a Python `http.server` handler is not guaranteed by a standard and must be measured on the target machine/browser.
- The repository does not yet expose a pointer signal, so the final producer rate and exact click/press semantics depend on the tracking implementation covered by the repository investigation.

## Coordinate mapping and smooth rendering

### Takeaway
Keep the shared tracker output normalized and render only the newest sample once per browser repaint. Map it to viewport CSS pixels, clamp it, position one fixed overlay using a CSS transform, and do calibration/filtering upstream so the browser prototype and eventual Linux cursor behave identically.

### Cited Findings
- `requestAnimationFrame()` invokes code before the next repaint and normally follows the display refresh rate; browsers usually pause it in background tabs. — [MDN: `requestAnimationFrame`](https://developer.mozilla.org/en-US/docs/Web/API/Window/requestAnimationFrame); [MDN: Page Visibility API](https://developer.mozilla.org/en-US/docs/Web/API/Page_Visibility_API)
- `document.elementFromPoint(x, y)` consumes viewport-relative coordinates and ignores elements styled with `pointer-events: none`. — [MDN: `elementFromPoint`](https://developer.mozilla.org/en-US/docs/Web/API/Document/elementFromPoint)
- CSS `pointer-events: none` makes pointer targeting pass through an overlay to the element underneath. — [MDN: CSS `pointer-events`](https://developer.mozilla.org/en-US/docs/Web/CSS/Reference/Properties/pointer-events)
- CSS transforms are the appropriate mechanism for changing visual position without repeatedly laying out via `top`/`left`; transform animation can be composited efficiently. — [MDN: Performance fundamentals](https://developer.mozilla.org/en-US/docs/Web/Performance/Guides/Fundamentals); [MDN: CSS and JavaScript animation performance](https://developer.mozilla.org/en-US/docs/Web/Performance/Guides/CSS_JavaScript_animation_performance)
- On zoomable/mobile layouts, the visual viewport has its own CSS-pixel width, height, offsets, and scale, while desktop layouts generally do not require that extra model. — [MDN: VisualViewport](https://developer.mozilla.org/en-US/docs/Web/API/VisualViewport)

### Inferences
- For the initial Linux desktop demo, compute `clientX = clamp(x, 0, 1) * document.documentElement.clientWidth` and similarly for height. Use the same client coordinates for `elementFromPoint`. Do not add mobile pinch-zoom handling until mobile is in scope.
- Style the synthetic cursor as `position: fixed; left: 0; top: 0; pointer-events: none; aria-hidden: true`, center it visually, and update only its `transform` inside a perpetual `requestAnimationFrame` loop. The SSE handler should only validate/store the latest sample.
- Start with no browser-side smoothing or prediction: both add lag or behavior that the later Linux backend would not share. Put calibration, dead zones, and any time-based smoothing in the native normalized-pointer stage; the page remains a thin output adapter.
- On `visibilitychange` to hidden, or when no valid sample arrives within a measured timeout, hide the overlay, clear hover state, and disarm activation. On resume/reconnect, require a fresh unpressed sample before accepting a press edge.

### Gaps
- The correct calibration transform, jitter filter, stale timeout, and camera-to-display mapping cannot be selected from browser standards; they require real hardware measurements.
- Multi-monitor mapping is intentionally absent from this page stage because a browser viewport is one rectangular target. It belongs to the later Linux output backend.

## Hover and activation semantics with accessibility

### Takeaway
Treat the overlay as an application-specific assistive pointer, not as a fake hardware mouse. Hit-test the owned page, expose hover with an explicit CSS class, and on a deliberate press edge focus then call `click()` on a narrow set of supported native controls; do not manufacture a stream of pointer events.

### Cited Findings
- The Pointer Events specification states that `PointerEventInit` constructs untrusted synthetic pointer events. — [W3C Pointer Events 3](https://www.w3.org/TR/pointerevents3/)
- `HTMLElement.click()` fires an element's click event unless the element is disabled, and HTML activation behavior runs for synthetic activations such as `element.click()`. — [MDN: `HTMLElement.click()`](https://developer.mozilla.org/en-US/docs/Web/API/HTMLElement/click); [WHATWG HTML: Input activation](https://html.spec.whatwg.org/multipage/input.html)
- The HTML standard says `click()` fires a synthetic click with the not-trusted flag and does not itself focus the element. — [WHATWG HTML: Activation behavior](https://html.spec.whatwg.org/multipage/interaction.html#activation-behavior-of-elements); [WHATWG HTML: Focus](https://html.spec.whatwg.org/multipage/interaction.html)
- `HTMLElement.focus({preventScroll: true})` can focus an element without scrolling it, and visible focus indication can be requested where supported. — [MDN: `HTMLElement.focus()`](https://developer.mozilla.org/en-US/docs/Web/API/HTMLElement/focus)
- Browser features including popups, fullscreen, pointer lock, file pickers, clipboard access, and device selectors require trusted transient user activation; synthetic events do not create it. — [MDN: User activation](https://developer.mozilla.org/en-US/docs/Web/Security/Defenses/User_activation)

### Inferences
- The first demo should support ordinary in-page actions only: native `button` and `a[href]` elements plus an explicit `[data-eyehand-activate]` opt-in for custom controls. Inputs requiring text entry, drag, selection, scrolling, context menus, file pickers, and activation-gated APIs are separate features.
- Each animation frame, hit-test below the overlay and apply one `.eyehand-hover` class to the nearest supported control. This gives the owned page a visible hover state without pretending the browser's real hardware `:hover` state changed.
- On one validated false-to-true press edge, call `target.focus({preventScroll: true})` and then `target.click()`. Latch until a subsequent release so noisy/repeated samples cannot double-activate. Disabled controls and stale/disconnected state must not activate.
- Keep the physical cursor visible during the prototype, show connection/armed state, provide a large physical-keyboard emergency disarm, and display press feedback before expanding supported actions. These are safety requirements for validating unintended activations, not optional polish.

### Gaps
- Synthetic activation behavior of complex native controls can vary and some actions are deliberately user-activation-gated; the initial acceptance surface should therefore be the project's own button/link test page.
- No gesture or dwell definition exists yet. The page should consume an upstream `pressed` state rather than inventing one, so the same intent detector can later drive Linux.

## Browser security boundary and arbitrary-tab control

### Takeaway
The page can draw and operate only its own DOM. It cannot move the real cursor, operate browser chrome, create trusted input, or reach arbitrary third-party page DOM; that later capability belongs to an OS input backend, not a stronger browser prototype.

### Cited Findings
- Same-origin policy restricts script interaction with documents from other origins and gives cross-origin `Window` references only limited access. — [MDN: Same-origin policy](https://developer.mozilla.org/en-US/docs/Web/Security/Defenses/Same-origin_policy)
- Pointer Lock provides mouse movement deltas, hides/locks the existing pointer, and requires an engagement gesture in relevant cases; in locked state there is no movable cursor and absolute mouse coordinates remain constant. It is not an API for setting an absolute cursor position. — [MDN: Pointer Lock API](https://developer.mozilla.org/en-US/docs/Web/API/Pointer_Lock_API)
- Only trusted input events establish the user-activation state required by sensitive browser APIs. — [MDN: User activation](https://developer.mozilla.org/en-US/docs/Web/Security/Defenses/User_activation)
- Cross-origin reads are blocked by default, and CORS should allow the minimum possible set of origins when cross-origin access is truly required. — [MDN: CORS configuration](https://developer.mozilla.org/en-US/docs/Web/Security/Practical_implementation_guides/CORS)

### Inferences
- Do not build an extension, automation driver, Pointer Lock integration, or injected content script for stage one. None is needed to prove coordinate quality and activation reliability on an owned page, and none would move the Linux system pointer.
- Bind only to `127.0.0.1`, serve page and stream from the same origin, send no permissive CORS header, accept only expected paths/methods, and validate all incoming sample fields before using them. If browser-to-native mutation is later added, add an unguessable session token and strict Origin checking.
- The clean architecture boundary is a normalized pointer/intent source feeding interchangeable sinks: first an owned-page/SSE sink, later a Linux input sink. The browser stage is an observable safety harness, not a stepping stone through increasingly invasive browser privileges.

### Gaps
- Browser extensions could manipulate permitted page DOM with explicit host permissions, but arbitrary-tab control was outside this assignment and does not help validate the eventual OS cursor backend.
- Exact Linux/Wayland/X11 injection permissions and APIs are covered by the Linux researcher, not this browser analysis.

## Acceptance tests and latency measurement

### Takeaway
Gate the prototype on correctness and fail-safe behavior first, then measure transport/render delay and physical motion-to-photon latency. Avoid claiming end-to-end latency from unsynchronized Python and browser monotonic clocks.

### Cited Findings
- `performance.now()` is a high-resolution monotonic clock, relative to the browser's `Performance.timeOrigin`, and is not subject to wall-clock adjustments. — [MDN: High precision timing](https://developer.mozilla.org/en-US/docs/Web/API/Performance_API/High_precision_timing)
- `requestAnimationFrame()` supplies a browser timeline timestamp and schedules work for the next repaint. — [MDN: `requestAnimationFrame`](https://developer.mozilla.org/en-US/docs/Web/API/Window/requestAnimationFrame)
- Background tabs usually stop receiving animation-frame callbacks, so visibility must be part of test setup and stale-state handling. — [MDN: Page Visibility API](https://developer.mozilla.org/en-US/docs/Web/API/Page_Visibility_API)

### Inferences
- Minimal automated browser logic tests: reject NaN/infinite/out-of-range samples; clamp normalized edges; ignore non-increasing sequence IDs; latest sample wins when several arrive before a frame; hover transitions clear the prior target; one press edge produces exactly one activation; disabled/unsupported elements do nothing; release rearms; disconnect/hidden/stale state clears hover and disarms.
- Minimal manual test page: a full-viewport grid with corner/edge/center targets, adjacent small targets, one disabled target, and counters for hover/activation. Success means every screen edge is reachable, the overlay never blocks hit-testing, targets activate only while visibly selected, and a stopped tracker cannot activate anything.
- Instrument `receiveTime = performance.now()` in the SSE handler and record the next animation-frame timestamp that paints that sequence. Report median/p95 receive-to-paint and dropped/superseded sample counts. At 60 Hz, use one refresh interval (about 16.7 ms) as the initial p95 browser-stage budget, then adjust for the actual display refresh rate.
- Measure native production-to-SSE-write separately with Python's monotonic clock. Do not subtract a Python monotonic timestamp from `performance.now()` because their time origins are unrelated.
- Measure true camera/gesture-to-visible-cursor latency with a high-frame-rate external camera or a synchronized on-screen/LED marker. That captures camera exposure, inference, transport, browser scheduling, display scanout, and avoids clock-alignment assumptions.
- Exit criterion for advancing to Linux control: stable calibrated normalized coordinates across the full viewport; no repeated or stale activations during an extended hands-on run; explicit disarm works every time; browser receive-to-paint meets the one-frame p95 budget while visible; and physical end-to-end latency/jitter is recorded as a baseline rather than guessed.

### Gaps
- No standards source defines an acceptable latency or accuracy threshold for this specific assistive eye-to-fingertip cursor. The proposed one-frame browser budget is an engineering acceptance criterion, not a published human-factors limit.
- End-to-end latency, spatial error, click false-positive rate, and filter tuning require the actual cameras, display, user, and target browser.
