const COLORS = {
  "Not Clickbait":  "#34c759",
  "Good Clickbait": "#ff9f0a",
  "Bad Clickbait":  "#ff3b30"
};

const BG_COLORS = {
  "Not Clickbait":  "linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%)",
  "Good Clickbait": "linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%)",
  "Bad Clickbait":  "linear-gradient(135deg, #fff1f0 0%, #ffe4e1 100%)"
};

const BORDER_COLORS = {
  "Not Clickbait":  "rgba(52,199,89,0.25)",
  "Good Clickbait": "rgba(255,159,10,0.25)",
  "Bad Clickbait":  "rgba(255,59,48,0.25)"
};

function fillExample(url) {
  document.getElementById("urlInput").value = url;
  document.getElementById("urlInput").focus();
}

async function analyze() {
  const url    = document.getElementById("urlInput").value.trim();
  const btn    = document.getElementById("analyzeBtn");
  const btnTxt = document.getElementById("btnText");
  const spinner= document.getElementById("btnSpinner");
  const errMsg = document.getElementById("errorMsg");

  if (!url) { showError("Please paste a YouTube URL first."); return; }

  btn.disabled = true;
  btnTxt.textContent = "Analyzing...";
  spinner.classList.remove("hidden");
  errMsg.classList.add("hidden");
  document.getElementById("resultSection").classList.add("hidden");

  try {
    const res  = await fetch("/predict", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ url })
    });
    const data = await res.json();
    if (data.error) { showError(data.error); return; }
    renderResult(data);
  } catch(e) {
    showError("Network error — is the Flask server running?");
  } finally {
    btn.disabled = false;
    btnTxt.textContent = "Analyze";
    spinner.classList.add("hidden");
  }
}

function renderResult(d) {
  const label = d.label;
  const color = COLORS[label];

  // Verdict hero
  const vh = document.getElementById("verdictHero");
  vh.style.background   = BG_COLORS[label];
  vh.style.borderColor  = BORDER_COLORS[label];
  document.getElementById("verdictIcon").textContent   = d.icon;
  document.getElementById("verdictLabel").textContent  = d.label;
  document.getElementById("verdictLabel").style.color  = color;
  document.getElementById("verdictDesc").textContent   = d.description;
  document.getElementById("confidenceVal").textContent = d.confidence + "%";
  document.getElementById("confidenceVal").style.color = color;

  // Video info
  document.getElementById("thumbnail").src          = d.thumbnail_url;
  document.getElementById("videoTitle").textContent = d.title;
  document.getElementById("videoChannel").textContent = d.channel;
  document.getElementById("viewCount").textContent  = "👁 " + d.view_count;
  document.getElementById("likeCount").textContent  = "👍 " + d.like_count;
  document.getElementById("publishedAt").textContent = "📅 " + d.published_at;
  document.getElementById("transcriptBadge").textContent =
    d.has_transcript ? "📝 Transcript available" : "❌ No transcript";
  document.getElementById("commentsBadge").textContent =
    d.has_comments ? "💬 Comments available" : "❌ No comments";

  // Prob bars
  const container = document.getElementById("probBars");
  container.innerHTML = "";
  for (const [name, pct] of Object.entries(d.probabilities)) {
    const c = COLORS[name] || "#888";
    container.innerHTML += `
      <div class="prob-row">
        <div class="prob-top">
          <span class="prob-name">${name}</span>
          <span class="prob-pct" style="color:${c}">${pct}%</span>
        </div>
        <div class="prob-track">
          <div class="prob-fill" style="width:0%;background:${c}"
               data-width="${pct}"></div>
        </div>
      </div>`;
  }
  // Animate bars after render
  requestAnimationFrame(() => {
    document.querySelectorAll(".prob-fill").forEach(el => {
      el.style.width = el.dataset.width + "%";
    });
  });

  // Similarity ring
  const simPct = Math.max(0, Math.min(100, d.similarity));
  const circumference = 251.2;
  const offset = circumference - (simPct / 100) * circumference;
  const ring   = document.getElementById("simRing");
  ring.style.strokeDashoffset = circumference; // reset
  ring.style.stroke = simPct > 30 ? "#34c759" : simPct > 15 ? "#ff9f0a" : "#ff3b30";
  requestAnimationFrame(() => {
    ring.style.strokeDashoffset = offset;
  });
  document.getElementById("simPct").textContent = simPct + "%";
  document.getElementById("simDesc").textContent =
    simPct > 30 ? "Strong match — title aligns well with content." :
    simPct > 15 ? "Partial match — some alignment between title and content." :
                  "Low match — title may not reflect actual content.";

  // Transcript
  if (d.transcript) {
    document.getElementById("transcriptText").textContent = d.transcript;
    document.getElementById("transcriptCard").style.display = "block";
  } else {
    document.getElementById("transcriptCard").style.display = "none";
  }

  document.getElementById("resultSection").classList.remove("hidden");
  setTimeout(() => {
    document.getElementById("resultSection").scrollIntoView({ behavior: "smooth", block: "start" });
  }, 100);
}

function showError(msg) {
  const el = document.getElementById("errorMsg");
  el.textContent = msg;
  el.classList.remove("hidden");
}

document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("urlInput").addEventListener("keydown", e => {
    if (e.key === "Enter") analyze();
  });
});