// DOM elements
const generationForm = document.getElementById("generationForm");
const promptInput = document.getElementById("prompt");
const numStepsInput = document.getElementById("numSteps");
const stepsDisplay = document.getElementById("stepsDisplay");
const generateBtn = document.getElementById("generateBtn");
const resolutionSelect = document.getElementById("resolution");
const charCount = document.getElementById("charCount");

const preview = document.getElementById("preview");
const loadingSpinner = document.getElementById("loadingSpinner");
const downloadSection = document.getElementById("downloadSection");
const downloadBtn = document.getElementById("downloadBtn");
const newGenerationBtn = document.getElementById("newGenerationBtn");
const errorMessage = document.getElementById("errorMessage");
const errorText = document.getElementById("errorText");
const systemInfoDiv = document.getElementById("systemInfo");

let currentSessionId = null;
let currentImageData = null;
let statusCheckInterval = null;

// Format system info for display
function formatSystemInfo(systemInfo) {
  const info = [
    { label: "Python Version", value: systemInfo.python_version },
    {
      label: "Platform",
      value: `${systemInfo.platform} ${systemInfo.platform_release}`,
    },
    {
      label: "CPU Cores",
      value: `${systemInfo.cpu_count} (${systemInfo.cpu_count_logical} logical)`,
    },
    { label: "CPU Frequency", value: `${systemInfo.cpu_freq} MHz` },
    { label: "RAM Total", value: `${systemInfo.ram_total_gb} GB` },
    { label: "RAM Available", value: `${systemInfo.ram_available_gb} GB` },
    { label: "PyTorch Version", value: systemInfo.torch_version },
    {
      label: "CUDA Available",
      value: systemInfo.cuda_available ? "✓ Yes" : "✗ No",
      cuda: systemInfo.cuda_available,
    },
  ];

  if (systemInfo.cuda_available) {
    info.push({ label: "CUDA Version", value: systemInfo.cuda_version });
    info.push({ label: "cuDNN Version", value: systemInfo.cudnn_version });
    info.push({ label: "GPU Devices", value: systemInfo.cuda_device_count });
    info.push({ label: "Current GPU", value: systemInfo.cuda_device_current });
    info.push({
      label: "GPU Memory",
      value: `${systemInfo.gpu_memory_total_gb} GB`,
    });
  }

  return info;
}

// Display system information
function displaySystemInfo(systemInfo) {
  const infoItems = formatSystemInfo(systemInfo);
  let html = "";

  infoItems.forEach((item) => {
    const valueClass =
      item.cuda !== undefined
        ? item.cuda
          ? "cuda-enabled"
          : "cuda-disabled"
        : "";
    html += `
      <div class="info-item">
        <span class="info-label">${item.label}</span>
        <span class="info-value ${valueClass}">${item.value}</span>
      </div>
    `;
  });

  systemInfoDiv.innerHTML = html;
}

// Character counter
promptInput.addEventListener("input", () => {
  charCount.textContent = promptInput.value.length;
});

// Steps display
numStepsInput.addEventListener("input", () => {
  stepsDisplay.textContent = numStepsInput.value;
});

// Form submission
generationForm.addEventListener("submit", async (e) => {
  e.preventDefault();

  if (!promptInput.value.trim()) {
    showError("Please enter a prompt");
    return;
  }

  await generateImage();
});

async function generateImage() {
  try {
    // Parse resolution
    const [width, height] = resolutionSelect.value.split("x").map(Number);

    const payload = {
      prompt: promptInput.value.trim(),
      height: height,
      width: width,
      num_steps: parseInt(numStepsInput.value),
      seed: parseInt(document.getElementById("seed").value),
    };

    // Show loading state
    showLoading();
    hideError();
    hideDownloadSection();

    // Send request
    const response = await fetch("/api/generate", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const data = await response.json();
      throw new Error(data.error || "Failed to generate image");
    }

    const data = await response.json();
    currentSessionId = data.session_id;

    // Start polling for status
    startStatusPolling();
  } catch (error) {
    console.error("Generation error:", error);
    showError(error.message);
    hideLoading();
  }
}

function startStatusPolling() {
  // Clear any existing interval
  if (statusCheckInterval) {
    clearInterval(statusCheckInterval);
  }

  // Check status immediately
  checkGenerationStatus();

  // Then poll every 1 second
  statusCheckInterval = setInterval(checkGenerationStatus, 1000);
}

async function checkGenerationStatus() {
  if (!currentSessionId) return;

  try {
    const response = await fetch(`/api/status/${currentSessionId}`);

    if (!response.ok) {
      throw new Error("Failed to check status");
    }

    const data = await response.json();

    if (data.status === "generating") {
      updateProgress(data.progress || 0);
    } else if (data.status === "completed") {
      clearInterval(statusCheckInterval);
      displayImage(data.image);
      hideLoading();
      showDownloadSection();
    } else if (data.status === "error") {
      clearInterval(statusCheckInterval);
      showError(data.error || "Unknown error occurred");
      hideLoading();
    }
  } catch (error) {
    console.error("Status check error:", error);
    // Don't show error for network issues, just log
  }
}

function displayImage(imageData) {
  currentImageData = imageData;
  preview.innerHTML = `<img src="data:image/png;base64,${imageData}" alt="Generated image">`;
}

function showLoading() {
  loadingSpinner.style.display = "flex";
  generateBtn.disabled = true;
}

function hideLoading() {
  loadingSpinner.style.display = "none";
  generateBtn.disabled = false;
}

function showDownloadSection() {
  downloadSection.style.display = "flex";
}

function hideDownloadSection() {
  downloadSection.style.display = "none";
}

function showError(message) {
  errorText.textContent = message;
  errorMessage.style.display = "block";
}

function hideError() {
  errorMessage.style.display = "none";
}

function updateProgress(progress) {
  const progressInfo = document.getElementById("progressInfo");
  progressInfo.textContent = `Progress: ${progress}%`;
}

// Download button
downloadBtn.addEventListener("click", () => {
  if (!currentImageData) return;

  const link = document.createElement("a");
  link.href = `data:image/png;base64,${currentImageData}`;
  link.download = `generated-${Date.now()}.png`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
});

// New generation button
newGenerationBtn.addEventListener("click", () => {
  preview.innerHTML =
    '<div class="placeholder"><p>Generated image will appear here</p></div>';
  hideDownloadSection();
  promptInput.focus();
  currentSessionId = null;
  currentImageData = null;
});

// Helper function to fill prompt from example
function fillPrompt(text) {
  promptInput.value = text;
  charCount.textContent = text.length;
  promptInput.focus();
}

// Initial health check
async function checkHealth() {
  try {
    const response = await fetch("/api/health");
    const data = await response.json();
    console.log("Health check:", data);

    if (data.system_info) {
      displaySystemInfo(data.system_info);
    }

    if (data.status !== "ok" && data.status !== "loading") {
      console.warn("Service not ready");
    }
  } catch (error) {
    console.error("Health check failed:", error);
    systemInfoDiv.innerHTML =
      '<p class="loading-text">Failed to load system information</p>';
  }
}

// Check health on page load
document.addEventListener("DOMContentLoaded", checkHealth);
