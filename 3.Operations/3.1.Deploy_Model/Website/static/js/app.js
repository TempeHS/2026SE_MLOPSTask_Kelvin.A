if ("serviceWorker" in navigator) {
  window.addEventListener("load", function () {
    navigator.serviceWorker
      .register("/static/js/serviceWorker.js")
      .then((res) => console.log("service worker registered"))
      .catch((err) => console.log("service worker not registered", err));
  });
}

// Clear cache when tab is closed
window.addEventListener("beforeunload", function () {
  // Clear sessionStorage
  sessionStorage.clear();

  // Send message to service worker to clear cache
  if (navigator.serviceWorker.controller) {
    navigator.serviceWorker.controller.postMessage({
      type: "CLEAR_CACHE",
    });
  }

  // Clear all caches
  if ("caches" in window) {
    caches.keys().then(function (cacheNames) {
      cacheNames.forEach(function (cacheName) {
        caches.delete(cacheName);
      });
    });
  }
});

// Also clear on visibility change (tab close/switch)
window.addEventListener("visibilitychange", function () {
  if (document.visibilityState === "hidden") {
    // Clear sessionStorage when tab is hidden
    sessionStorage.clear();
  }
});

// This script toggles the active class and aria-current attribute on the nav links
document.addEventListener("DOMContentLoaded", function () {
  const navLinks = document.querySelectorAll(".nav-link");
  const currentUrl = window.location.pathname;

  navLinks.forEach((link) => {
    const linkUrl = link.getAttribute("href");
    if (linkUrl === currentUrl) {
      link.classList.add("active");
      link.setAttribute("aria-current", "page");
    } else {
      link.classList.remove("active");
      link.removeAttribute("aria-current");
    }
  });

  // Toggle password visibility
  const togglePasswordBtn = document.getElementById("togglePassword");
  const passwordInput = document.getElementById("exampleInputPassword");

  if (togglePasswordBtn && passwordInput) {
    togglePasswordBtn.addEventListener("click", function () {
      const isPassword = passwordInput.type === "password";
      passwordInput.type = isPassword ? "text" : "password";
      togglePasswordBtn.innerHTML = isPassword
        ? '<i class="bi bi-eye-slash"></i> Hide'
        : '<i class="bi bi-eye"></i> Show';
    });
  }
});
