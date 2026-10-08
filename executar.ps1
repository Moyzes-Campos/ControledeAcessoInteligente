param(
    [Parameter(Mandatory = $true)]
    [string]$InputVideo,
    [string]$Output = "outputs\execucao",
    [ValidateSet(-90, 0, 90, 180)]
    [int]$Rotate = 0,
    [ValidateSet("scrfd", "yunet")]
    [string]$FaceDetector = "scrfd",
    [ValidateSet("auto", "cuda", "tensorrt")]
    [string]$FaceRuntime = "auto",
    [string]$FaceEngine = "models\insightface_buffalo_l\det_10g_fp16.engine",
    [ValidateSet("arcface", "sface")]
    [string]$FaceRecognizer = "arcface",
    [string]$FaceRecognizerEngine = "models\insightface_buffalo_l\w600k_r50_fp16.engine",
    [ValidateRange(-1.0, 1.0)]
    [Nullable[double]]$FaceThreshold = $null,
    [ValidateRange(0.0, 1.0)]
    [double]$FaceDetectScore = 0.50,
    [ValidateRange(0, 2147483647)]
    [int]$FaceRecheckInterval = 0,
    [ValidateRange(1, 2147483647)]
    [int]$FaceInterval = 3,
    [ValidateRange(0.0, 3600.0)]
    [double]$FaceRetrySeconds = 0.5,
    [ValidateRange(1, 2147483647)]
    [int]$FaceMaxTracks = 1,
    [ValidateSet(-90, 0, 90, 180)]
    [int[]]$FaceRotations = @(0, 90, 180, -90),
    [switch]$FaceSync,
    [ValidateRange(0.0, 1000.0)]
    [double]$MaxFps = 0.0,
    [ValidateRange(0, 16384)]
    [int]$PreviewMaxSide = 1280,
    [ValidateRange(0, 64)]
    [int]$CaptureThreads = 1,
    [string]$Database = "data\acessos.sqlite3",
    [string]$LineConfig = "outputs\linha\linha.json",
    [switch]$InvertLine,
    [string]$EspUrl = "http://192.168.3.1",
    [switch]$NoEsp,
    [ValidateRange(0.1, 3600.0)]
    [double]$AlarmSeconds = 3.0,
    [switch]$Show,
    [switch]$SaveVideo
)

$projectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path (Split-Path -Parent $projectDir) ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Ambiente Python nao encontrado: $python"
}

$arguments = @(
    (Join-Path $projectDir "main.py"),
    "--input", $InputVideo,
    "--output", (Join-Path $projectDir $Output),
    "--model", (Join-Path (Split-Path -Parent $projectDir) "models\yolo_v26\.engine\yolo26x_fp16.engine"),
    "--rotate", $Rotate,
    "--face-detector", $FaceDetector,
    "--face-runtime", $FaceRuntime,
    "--face-engine", $(if ([IO.Path]::IsPathRooted($FaceEngine)) { $FaceEngine } else { Join-Path $projectDir $FaceEngine }),
    "--face-recognizer", $FaceRecognizer,
    "--face-recognizer-engine", $(if ([IO.Path]::IsPathRooted($FaceRecognizerEngine)) { $FaceRecognizerEngine } else { Join-Path $projectDir $FaceRecognizerEngine }),
    "--face-detect-score", $FaceDetectScore,
    "--face-recheck-interval", $FaceRecheckInterval,
    "--face-interval", $FaceInterval,
    "--face-retry-seconds", $FaceRetrySeconds.ToString([Globalization.CultureInfo]::InvariantCulture),
    "--face-max-tracks", $FaceMaxTracks,
    "--max-fps", $MaxFps.ToString([Globalization.CultureInfo]::InvariantCulture),
    "--preview-max-side", $PreviewMaxSide,
    "--capture-threads", $CaptureThreads,
    "--database", $(if ([IO.Path]::IsPathRooted($Database)) { $Database } else { Join-Path $projectDir $Database }),
    "--line-config", $(if ([IO.Path]::IsPathRooted($LineConfig)) { $LineConfig } else { Join-Path $projectDir $LineConfig }),
    "--esp-url", $EspUrl,
    "--alarm-seconds", $AlarmSeconds
)
if ($null -ne $FaceThreshold) {
    $arguments += "--face-threshold"
    $arguments += $FaceThreshold.Value.ToString([Globalization.CultureInfo]::InvariantCulture)
}
$arguments += "--face-rotations"
$arguments += $FaceRotations
if ($FaceSync) { $arguments += "--face-sync" }
if ($Show) { $arguments += "--show" }
if ($SaveVideo) { $arguments += "--save-video" }
if ($InvertLine) { $arguments += "--invert-line" }
if ($NoEsp) { $arguments += "--no-esp" }

& $python @arguments
exit $LASTEXITCODE

