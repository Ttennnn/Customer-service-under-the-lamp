/* Shared acceptance-page helpers: fetch, job controls, logs, formatting. */
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

  function mountJobControls(container, gate, onDone) {
    var job = gate.job;
    var controls = el("div", "job-controls");
    var runButton = el("button", "primary", "重跑 " + job);
    runButton.type = "button";
    var stopButton = el("button", "danger", "停止");
    stopButton.type = "button";
    stopButton.disabled = true;
    var status = el("span", "muted", "");
    var log = el("pre", "job-log", "");
    controls.appendChild(runButton);
    controls.appendChild(stopButton);
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

    function poll(myToken) {
      jsonFetch("/api/jobs/" + encodeURIComponent(job) + "/status")
        .then(function (data) {
          if (myToken !== token) return;
          log.textContent = data.output_tail || "";
          if (data.running) {
            status.className = "muted";
            status.textContent = "运行中…（开始于 " + fmtTime(data.started_at) + "）";
            runButton.disabled = true;
            stopButton.disabled = false;
            timer = setTimeout(function () { poll(myToken); }, 1200);
            return;
          }
          runButton.disabled = false;
          stopButton.disabled = true;
          if (data.exit_code === null || data.exit_code === undefined) {
            status.className = "muted";
            status.textContent = "未运行";
            return;
          }
          status.className = data.exit_code === 0 ? "muted" : "status error";
          status.textContent = "退出码 " + data.exit_code +
            (data.finished_at ? "（" + fmtTime(data.finished_at) + "）" : "");
          if (onDone) onDone(data);
        })
        .catch(function (err) {
          if (myToken !== token) return;
          status.className = "status error";
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
          status.className = "muted";
          status.textContent = "已发起…";
          runButton.disabled = true;
          stopButton.disabled = false;
          stopPolling();
          var myToken = token;
          poll(myToken);
        })
        .catch(function (err) {
          status.className = "status error";
          status.textContent = "发起失败：" + err.message;
        });
    });

    stopButton.addEventListener("click", function () {
      jsonFetch("/api/jobs/" + encodeURIComponent(job) + "/stop", {
        method: "POST"
      })
        .then(function (data) {
          status.className = "muted";
          status.textContent = data.stopped ? "已请求停止…" : "没有正在运行的作业";
        })
        .catch(function (err) {
          status.className = "status error";
          status.textContent = "停止失败：" + err.message;
        });
    });

    // Show the last known state/log without starting anything.
    var initial = token;
    jsonFetch("/api/jobs/" + encodeURIComponent(job) + "/status")
      .then(function (data) {
        if (initial !== token) return;
        log.textContent = data.output_tail || "";
        if (data.running) {
          status.textContent = "运行中…";
          stopButton.disabled = false;
          runButton.disabled = true;
          var myToken = token;
          poll(myToken);
        } else if (data.exit_code !== null && data.exit_code !== undefined) {
          status.textContent = "上次退出码 " + data.exit_code;
        } else {
          status.textContent = "未运行";
        }
      })
      .catch(function () { /* unknown jobs are reported when clicked */ });

    return controls;
  }

  window.Acceptance = {
    el: el,
    jsonFetch: jsonFetch,
    readError: readError,
    fmtTime: fmtTime,
    fmtNumber: fmtNumber,
    mountJobControls: mountJobControls,
    HEAVY: HEAVY
  };
})();
