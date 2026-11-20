/* Shared acceptance-page helpers: fetch, compact job controls, tabs, formatting. */
(function () {
  "use strict";

  var HEAVY = new Set(["ch10-corpus", "ch10-dataset", "ch10-train", "ch10-export"]);

  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }

  function readError(response) {
    return response.text().then(function (raw) {
      var detail = "HTTP " + response.status;
      try {
        var data = raw ? JSON.parse(raw) : null;
        if (data && typeof data.detail === "string") detail = data.detail;
      } catch (err) { /* keep default */ }
      return detail;
    });
  }

  function jsonFetch(url, options) {
    return fetch(url, options).then(function (response) {
      if (!response.ok) {
        return readError(response).then(function (detail) {
          throw new Error(detail);
        });
      }
      return response.json();
    });
  }

  function fmtTime(value) {
    if (!value) return "—";
    var date = new Date(value);
    if (isNaN(date.getTime())) return String(value);
    return date.toLocaleString("zh-CN", { hour12: false });
  }

  function fmtNumber(value, digits) {
    var n = Number(value);
    if (!Number.isFinite(n)) return "—";
    return n.toFixed(digits == null ? 3 : digits);
  }

  function detailText(detail, limit) {
    if (detail === null || detail === undefined) return "—";
    if (typeof detail !== "object") return String(detail);
    var parts = Object.keys(detail).map(function (key) {
      var value = detail[key];
      if (value === null || value === undefined) value = "—";
      if (typeof value === "object") value = JSON.stringify(value);
      return key + "=" + value;
    });
    var text = parts.join(" · ");
    var max = limit || 200;
    return text.length > max ? text.slice(0, max) + "…" : text;
  }

  function stateBadge(state) {
    var label = state === "pass" ? "通过" : (state === "fail" ? "未过线" : "未跑过");
    return el("span", "badge " + state, label);
  }

  function mountJobControls(container, gate, onDone) {
    var job = gate.job;
    var controls = el("div", "job-controls");
    var runButton = el("button", "primary small", "重跑");
    runButton.type = "button";
    var stopButton = el("button", "danger small", "停止");
    stopButton.type = "button";
    stopButton.disabled = true;
    var logButton = el("button", "small", "日志");
    logButton.type = "button";
    var status = el("span", "job-status", "");
    var log = el("pre", "job-log", "");
    controls.appendChild(runButton);
    controls.appendChild(stopButton);
    controls.appendChild(logButton);
    controls.appendChild(status);
    container.appendChild(controls);
    container.appendChild(log);

    var timer = null;
    var token = 0;

    function stopPolling() {
      token += 1;
      if (timer) {
        clearTimeout(timer);
        timer = null;
      }
    }

    logButton.addEventListener("click", function () {
      var open = log.classList.toggle("open");
      logButton.textContent = open ? "收起" : "日志";
    });

    function poll(myToken) {
      jsonFetch("/api/jobs/" + encodeURIComponent(job) + "/status")
        .then(function (data) {
          if (myToken !== token) return;
          log.textContent = data.output_tail || "";
          if (data.running) {
            status.textContent = "运行中…";
            runButton.disabled = true;
            stopButton.disabled = false;
            timer = setTimeout(function () { poll(myToken); }, 1200);
            return;
          }
          runButton.disabled = false;
          stopButton.disabled = true;
          if (data.exit_code === null || data.exit_code === undefined) {
            status.textContent = "未运行";
            return;
          }
          status.textContent = "退出码 " + data.exit_code;
          if (onDone) onDone(data);
        })
        .catch(function (err) {
          if (myToken !== token) return;
          status.textContent = "状态读取失败：" + err.message;
        });
    }

    runButton.addEventListener("click", function () {
      var prompt = HEAVY.has(job)
        ? "「" + job + "」是重作业，可能跑几分钟。继续？"
        : "重跑「" + job + "」？";
      if (!window.confirm(prompt)) return;
      jsonFetch("/api/jobs/" + encodeURIComponent(job) + "/run", {
        method: "POST"
      })
        .then(function () {
          status.textContent = "已发起…";
          runButton.disabled = true;
          stopButton.disabled = false;
          stopPolling();
          poll(token);
        })
        .catch(function (err) {
          status.textContent = "发起失败：" + err.message;
        });
    });

    stopButton.addEventListener("click", function () {
      jsonFetch("/api/jobs/" + encodeURIComponent(job) + "/stop", {
        method: "POST"
      })
        .then(function (data) {
          status.textContent = data.stopped ? "已请求停止…" : "没有正在运行的作业";
        })
        .catch(function (err) {
          status.textContent = "停止失败：" + err.message;
        });
    });

    // Quietly show the last known state; logs stay collapsed.
    var initial = token;
    jsonFetch("/api/jobs/" + encodeURIComponent(job) + "/status")
      .then(function (data) {
        if (initial !== token) return;
        log.textContent = data.output_tail || "";
        if (data.running) {
          status.textContent = "运行中…";
          stopButton.disabled = false;
          runButton.disabled = true;
          poll(token);
        } else if (data.exit_code !== null && data.exit_code !== undefined) {
          status.textContent = "上次退出码 " + data.exit_code;
        } else {
          status.textContent = "未运行";
        }
      })
      .catch(function () { /* unknown jobs surface when clicked */ });

    return controls;
  }

  function setupTabs() {
    var buttons = Array.prototype.slice.call(
      document.querySelectorAll(".tab[data-tab]")
    );
    if (!buttons.length) return;
    var panels = Array.prototype.slice.call(
      document.querySelectorAll(".tab-panel")
    );
    function activate(id) {
      buttons.forEach(function (button) {
        var active = button.getAttribute("data-tab") === id;
        button.classList.toggle("active", active);
        button.setAttribute("aria-selected", active ? "true" : "false");
      });
      panels.forEach(function (panel) {
        panel.hidden = panel.id !== "tab-" + id;
      });
    }
    buttons.forEach(function (button) {
      button.addEventListener("click", function () {
        activate(button.getAttribute("data-tab"));
      });
    });
    var current = buttons.filter(function (button) {
      return button.classList.contains("active");
    })[0] || buttons[0];
    activate(current.getAttribute("data-tab"));
  }

  window.Acceptance = {
    el: el,
    jsonFetch: jsonFetch,
    readError: readError,
    fmtTime: fmtTime,
    fmtNumber: fmtNumber,
    detailText: detailText,
    stateBadge: stateBadge,
    mountJobControls: mountJobControls,
    setupTabs: setupTabs,
    HEAVY: HEAVY
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", setupTabs);
  } else {
    setupTabs();
  }
})();
