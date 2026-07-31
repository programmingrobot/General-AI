const form = document.querySelector("#chatForm");
const input = document.querySelector("#messageInput");
const messages = document.querySelector("#messages");
const workerStatus = document.querySelector("#workerStatus");

const activeJobs = new Map();

function setMessageContent(row, content, image) {
  row.querySelector("p").textContent = content;
  const existingImage = row.querySelector("img");
  if (existingImage) {
    existingImage.remove();
  }
  if (image) {
    const generatedImage = document.createElement("img");
    generatedImage.className = "generated-image";
    generatedImage.alt = content || "Generated image";
    generatedImage.src = image;
    row.append(generatedImage);
  }
}

function addMessage(role, content, id, image) {
  const row = document.createElement("article");
  row.className = `message ${role}`;
  if (id) {
    row.dataset.jobId = id;
  }

  const label = document.createElement("strong");
  label.textContent = role === "user" ? "You" : "AI";

  const body = document.createElement("p");
  body.textContent = content;

  row.append(label, body);
  if (image) {
    const generatedImage = document.createElement("img");
    generatedImage.className = "generated-image";
    generatedImage.alt = content || "Generated image";
    generatedImage.src = image;
    row.append(generatedImage);
  }
  messages.append(row);
  messages.scrollTop = messages.scrollHeight;
  return row;
}

async function loadHistory() {
  const response = await fetch("/history");
  const payload = await response.json();
  messages.replaceChildren();
  for (const message of payload.messages) {
    addMessage(message.role, message.content, null, message.image);
  }
}

async function pollJob(id, row) {
  const startedAt = Date.now();
  const timer = setInterval(async () => {
    try {
      const response = await fetch(`/chat/${id}`);
      const job = await response.json();
      const waitingSeconds = Math.max(1, Math.round((Date.now() - startedAt) / 1000));

      if (job.status === "queued") {
        workerStatus.textContent = "Queued locally";
        setMessageContent(row, `Queued... ${waitingSeconds}s`);
      } else if (job.status === "processing") {
        workerStatus.textContent = "Qwen is thinking";
        setMessageContent(row, job.interim_response || `Thinking... ${waitingSeconds}s`);
      } else if (job.status === "generating_image") {
        workerStatus.textContent = "Generating image";
        setMessageContent(row, job.interim_response || `Generating image... ${waitingSeconds}s`);
      } else {
        clearInterval(timer);
        activeJobs.delete(id);
        workerStatus.textContent = activeJobs.size ? "Working" : "Ready";
        row.classList.toggle("error", job.status === "error");
        setMessageContent(row, job.error || job.response || "No response.", job.image);
      }
    } catch (error) {
      clearInterval(timer);
      activeJobs.delete(id);
      row.classList.add("error");
      setMessageContent(row, `Console error: ${error.message}`);
    }
  }, 500);
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = input.value.trim();
  if (!message) {
    return;
  }

  input.value = "";
  addMessage("user", message);
  const aiRow = addMessage("assistant", "Queued...");

  const response = await fetch("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message }),
  });
  const payload = await response.json();

  if (!response.ok) {
    aiRow.classList.add("error");
    setMessageContent(aiRow, payload.error || "Could not send message.");
    return;
  }

  activeJobs.set(payload.id, aiRow);
  pollJob(payload.id, aiRow);
});

loadHistory().catch(() => {
  workerStatus.textContent = "Could not load history";
});
