param(
    [ValidateRange(1, 65535)]
    [int]$Port = 8765,
    [switch]$NoBrowser,
    [switch]$NoWait
)

$ErrorActionPreference = "Stop"
$projectRoot = [System.IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$hostAddress = "127.0.0.1"
$baseUrl = "http://${hostAddress}:$Port"
$livenessUrl = "$baseUrl/health/live"
$logRoot = Join-Path $projectRoot "runtime\logs"
$ownedProcess = $null
$stopOwnedProcessOnExit = $false

function Write-Banner {
    Write-Host ""
    Write-Host "----------------------------------------" -ForegroundColor Cyan
    Write-Host "SceneMind-X 多模态视觉资产系统" -ForegroundColor Cyan
    Write-Host "SceneMind-X Multimodal Visual Asset System" -ForegroundColor Cyan
    Write-Host "----------------------------------------" -ForegroundColor Cyan
    Write-Host ""
}

function Write-Step(
    [int]$Number,
    [string]$Chinese,
    [string]$English
) {
    Write-Host "[$Number/5] $Chinese" -ForegroundColor Yellow
    Write-Host "      $English" -ForegroundColor DarkGray
}

function Write-FriendlyError([string]$Chinese, [string]$English) {
    Write-Host ""
    Write-Host "[SceneMind-X] $Chinese" -ForegroundColor Red
    Write-Host "[SceneMind-X] $English" -ForegroundColor Red
}

function Get-Liveness {
    try {
        $response = Invoke-RestMethod -Uri $livenessUrl -TimeoutSec 3
        if ($response.status -eq "ok" -and $response.liveness -eq "alive") {
            return $response
        }
    } catch {
        return $null
    }
    return $null
}

function Test-PortOpen {
    $client = [System.Net.Sockets.TcpClient]::new()
    try {
        $async = $client.BeginConnect($hostAddress, $Port, $null, $null)
        if (-not $async.AsyncWaitHandle.WaitOne(600)) {
            return $false
        }
        $client.EndConnect($async)
        return $true
    } catch {
        return $false
    } finally {
        $client.Dispose()
    }
}

function Open-SceneMindX {
    if ($NoBrowser -or $env:SCENEMINDX_START_NO_BROWSER -eq "1") {
        return
    }
    Start-Process $baseUrl | Out-Null
}

function Resolve-ProjectPython {
    $candidates = @(
        (Join-Path $projectRoot ".venv\Scripts\python.exe"),
        $env:SCENEMINDX_PYTHON
    )
    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path -LiteralPath $candidate -PathType Leaf)) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }
    if ($env:SCENEMINDX_START_DISABLE_SYSTEM_PYTHON -ne "1") {
        $systemPython = Get-Command python.exe -ErrorAction SilentlyContinue
        if ($systemPython) {
            return $systemPython.Source
        }
    }
    return $null
}

function Assert-PackageFiles {
    $required = @(
        "apps\api\main.py",
        "src\scenemindx",
        "prompts",
        "configs",
        "data\manifests\phase6_1_train_assets.jsonl",
        "data\manifests\phase6_1_val_assets.jsonl",
        "datasets\course_train",
        "datasets\course_val",
        "requirements.txt",
        "scripts\start_scenemindx_process.py"
    )
    $missing = @()
    foreach ($relativePath in $required) {
        $candidate = Join-Path $projectRoot $relativePath
        if (-not (Test-Path -LiteralPath $candidate)) {
            $missing += $relativePath
        }
    }
    if ($missing.Count -gt 0) {
        throw "课程包缺少必要文件 / Required package files are missing: $($missing -join ', ')"
    }
}

function Set-SceneMindXEnvironment {
    # Only the Python interpreter may come from another environment. All
    # business data, configuration, prompts, indexes and runtime state are
    # resolved inside this course-submission directory.
    $sourceRoot = Join-Path $projectRoot "src"
    $env:SCENEMINDX_PROJECT_ROOT = $projectRoot
    $env:PYTHONPATH = $sourceRoot
    $env:SCENEMINDX_RUN_ROOT = Join-Path $projectRoot "runtime"

    $trainRoot = Join-Path $projectRoot "datasets\course_train"
    $valRoot = Join-Path $projectRoot "datasets\course_val"
    $env:SCENEMINDX_DATASET_ROOT = $trainRoot
    $env:SCENEMINDX_SYSTEM_TRAIN_ROOT = $trainRoot
    $env:SCENEMINDX_SYSTEM_VAL_ROOT = $valRoot

    $env:SCENEMINDX_MANIFEST_PATH = Join-Path $projectRoot "data\manifests\gate1_d3_hard_train.jsonl"
    $env:SCENEMINDX_BAILIAN_CREDENTIALS_PATH = Join-Path $projectRoot ".secrets\bailian_credentials.csv"
    # Cloud is the recommended course-review path. Self-hosted remains
    # available only when the user explicitly supplies both remote endpoints;
    # the portable launcher must not silently assume a private SSH tunnel.
    $selfHostedConfigured = [bool](
        $env:SCENEMINDX_VLM_ENDPOINT -and
        $env:SCENEMINDX_E1_EMBEDDING_ENDPOINT
    )
    $env:SCENEMINDX_ENABLE_VLM = if ($selfHostedConfigured) { "1" } else { "0" }
    $env:SCENEMINDX_VLM_INLINE_IMAGES = if ($selfHostedConfigured) { "1" } else { "0" }
    $env:SCENEMINDX_ENABLE_EMBEDDING = "1"
    $env:SCENEMINDX_EMBEDDING_BACKEND = "deterministic_baseline"
    $env:SCENEMINDX_RETRIEVAL_BACKEND = if ($selfHostedConfigured) { "e1" } else { "r0" }
    $env:SCENEMINDX_RETRIEVAL_FALLBACK = "r0"
    $env:SCENEMINDX_E1_INDEX_ROOT = Join-Path $projectRoot "data\indexes\local_e1_2048\product"
    $env:SCENEMINDX_SYSTEM_TRAIN_ASSET_MANIFEST = Join-Path $projectRoot "data\manifests\phase6_1_train_assets.jsonl"
    $env:SCENEMINDX_SYSTEM_VAL_ASSET_MANIFEST = Join-Path $projectRoot "data\manifests\phase6_1_val_assets.jsonl"
    $env:SCENEMINDX_SYSTEM_LIBRARY_CATALOG = Join-Path $projectRoot "data\manifests\phase6_1_system_libraries.json"
    $env:SCENEMINDX_SYSTEM_THUMBNAIL_ROOT = Join-Path $projectRoot "data\cache\thumbnails\phase6_1"
    $env:SCENEMINDX_SYSTEM_TRAIN_ACTIVE_MANIFEST = Join-Path $projectRoot "data\manifests\phase6_1_train_active_manifest.jsonl"
    $env:SCENEMINDX_SYSTEM_VAL_ACTIVE_MANIFEST = Join-Path $projectRoot "data\manifests\phase6_1_val_active_manifest.jsonl"
    $env:SCENEMINDX_SYSTEM_E1_INDEX_ROOT = Join-Path $projectRoot "data\indexes\local_e1_2048"
    $env:SCENEMINDX_API_PORT = [string]$Port
}

function Stop-OwnedSceneMindX {
    if ($ownedProcess -and -not $ownedProcess.HasExited) {
        Write-Host ""
        Write-Host "[SceneMind-X] 正在关闭服务... / Stopping the service..." -ForegroundColor Yellow
        Stop-Process -Id $ownedProcess.Id -ErrorAction SilentlyContinue
        Wait-Process -Id $ownedProcess.Id -Timeout 10 -ErrorAction SilentlyContinue
        Write-Host "[SceneMind-X] 服务已关闭。 / Service stopped." -ForegroundColor Green
    }
}

try {
    Write-Banner

    Write-Step 1 "正在检查课程包文件..." "Checking course-package files..."
    Assert-PackageFiles
    Write-Host "      课程包路径 / Package root: $projectRoot" -ForegroundColor DarkGray

    Write-Step 2 "正在检查端口和已有服务..." "Checking the port and existing service..."
    $existing = Get-Liveness
    if ($existing) {
        Write-Host ""
        Write-Host "[SceneMind-X] 服务已经就绪。 / The service is already ready." -ForegroundColor Green
        Write-Host "浏览器地址 / Browser: $baseUrl" -ForegroundColor Cyan
        Open-SceneMindX
        exit 0
    }
    if (Test-PortOpen) {
        Write-FriendlyError "端口 $Port 已被其他程序或异常服务占用，未停止任何进程。" "Port $Port is already in use by another or unhealthy program. Nothing was stopped."
        exit 2
    }

    Write-Step 3 "正在检查 Python 环境与依赖..." "Checking Python and required packages..."
    $python = Resolve-ProjectPython
    if (-not $python) {
        Write-FriendlyError "未找到 Python。请安装 Python 3.11 或在 README 指引下设置 SCENEMINDX_PYTHON。" "Python was not found. Install Python 3.11 or set SCENEMINDX_PYTHON as described in README."
        exit 3
    }
    Set-Location -LiteralPath $projectRoot
    Set-SceneMindXEnvironment
    $previousErrorAction = $ErrorActionPreference
    $dependencyOutput = @()
    try {
        $ErrorActionPreference = "Continue"
        $dependencyOutput = & $python -c "import fastapi, uvicorn, scenemindx" 2>&1
        $dependencyExitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousErrorAction
    }
    if ($dependencyExitCode -ne 0) {
        Write-FriendlyError "当前 Python 缺少依赖。请在项目根目录运行：python -m pip install -r requirements.txt" "The selected Python is missing dependencies. Run: python -m pip install -r requirements.txt"
        if ($dependencyOutput) {
            Write-Host "      $($dependencyOutput | Select-Object -First 3 | Out-String)" -ForegroundColor DarkGray
        }
        exit 4
    }
    Write-Host "      Python: $python" -ForegroundColor DarkGray

    Write-Step 4 "正在启动 SceneMind-X 服务..." "Starting SceneMind-X services..."
    New-Item -ItemType Directory -Path $logRoot -Force | Out-Null
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $stdoutPath = Join-Path $logRoot "scenemindx_${Port}_${timestamp}_stdout.log"
    $stderrPath = Join-Path $logRoot "scenemindx_${Port}_${timestamp}_stderr.log"
    $processHelper = Join-Path $PSScriptRoot "start_scenemindx_process.py"
    $launchText = & $python $processHelper `
        --project-root $projectRoot `
        --python $python `
        --port $Port `
        --stdout $stdoutPath `
        --stderr $stderrPath
    if ($LASTEXITCODE -ne 0) {
        Write-FriendlyError "无法创建后端进程。" "The backend process could not be created."
        exit 5
    }
    $launch = $launchText | ConvertFrom-Json
    $ownedProcess = Get-Process -Id ([int]$launch.launcher_pid) -ErrorAction SilentlyContinue
    if (-not $ownedProcess) {
        Write-FriendlyError "后端启动后立即退出。错误日志：$stderrPath" "The backend exited immediately. Error log: $stderrPath"
        exit 5
    }
    $stopOwnedProcessOnExit = $true

    Write-Step 5 "正在等待 Web 服务真实就绪..." "Waiting for the web service health check..."
    $deadline = [DateTime]::UtcNow.AddSeconds(90)
    $health = $null
    do {
        Start-Sleep -Milliseconds 500
        $ownedProcess.Refresh()
        if ($ownedProcess.HasExited) {
            Write-FriendlyError "后端在健康检查通过前退出。错误日志：$stderrPath" "The backend exited before health became ready. Error log: $stderrPath"
            exit 5
        }
        $health = Get-Liveness
    } while (-not $health -and [DateTime]::UtcNow -lt $deadline)

    if (-not $health) {
        Write-FriendlyError "等待 90 秒后仍未就绪。错误日志：$stderrPath" "The service was not ready after 90 seconds. Error log: $stderrPath"
        exit 6
    }

    $pidPath = Join-Path $logRoot "scenemindx_${Port}.pid"
    Set-Content -LiteralPath $pidPath -Value ([string]$ownedProcess.Id) -Encoding ascii
    Write-Host ""
    Write-Host "[SceneMind-X] 系统已就绪。 / SceneMind-X is ready." -ForegroundColor Green
    Write-Host "浏览器地址 / Browser: $baseUrl" -ForegroundColor Cyan
    Write-Host "运行日志 / Logs: $stdoutPath ; $stderrPath" -ForegroundColor DarkGray
    Write-Host "----------------------------------------" -ForegroundColor Cyan
    Open-SceneMindX

    if ($NoWait -or $env:SCENEMINDX_START_NO_WAIT -eq "1") {
        $stopOwnedProcessOnExit = $false
        exit 0
    }

    Write-Host ""
    Write-Host "保持此窗口打开。按 Enter 或 Ctrl+C 可安全关闭本次启动的服务。" -ForegroundColor Yellow
    Write-Host "Keep this window open. Press Enter or Ctrl+C to stop this service." -ForegroundColor Yellow
    [void](Read-Host)
    Stop-OwnedSceneMindX
    $stopOwnedProcessOnExit = $false
    exit 0
} catch {
    Write-FriendlyError $_.Exception.Message "Startup failed. See the message above and runtime/logs."
    exit 10
} finally {
    if ($stopOwnedProcessOnExit) {
        Stop-OwnedSceneMindX
    }
}
