const assets = [
  "/static/css/style.css",
  "/static/css/bootstrap.min.css",
  "/static/js/bootstrap.bundle.min.js",
  "/static/js/app.js",
  "/static/images/logo.png",
  "/static/images/favicon.jpg",
  "/static/icons/icon-128x128.png",
  "/static/icons/icon-192x192.png",
  "/static/icons/icon-384x384.png",
  "/static/icons/icon-512x512.png",
  "/static/icons/desktop_screenshot.png",
  "/static/icons/mobile_screenshot.png",
  "/offline.html",
];

// Don't cache these pages
const NO_CACHE_URLS = [
  "/",
  "/index.html",
  "/Login.html",
  "/Signup.html",
  "/verify_2fa",
  "/privacy.html",
];

const CATALOGUE_ASSETS = "catalogue-assets";

self.addEventListener("install", (installEvt) => {
  installEvt.waitUntil(
    caches
      .open(CATALOGUE_ASSETS)
      .then((cache) => {
        console.log(cache);
        cache.addAll(assets);
      })
      .then(self.skipWaiting())
      .catch((e) => {
        console.log(e);
      })
  );
});

self.addEventListener("activate", function (evt) {
  evt.waitUntil(
    caches
      .keys()
      .then((keyList) => {
        return Promise.all(
          keyList.map((key) => {
            if (key === CATALOGUE_ASSETS) {
              console.log("Removed old cache from", key);
              return caches.delete(key);
            }
          })
        );
      })
      .then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", function (evt) {
  const url = new URL(evt.request.url);
  const pathname = url.pathname;

  // Don't cache sensitive pages
  if (NO_CACHE_URLS.includes(pathname)) {
    evt.respondWith(
      fetch(evt.request, { cache: "no-store" }).catch(() => {
        if (evt.request.mode === "navigate") {
          return caches.match("/offline.html");
        }
        return new Response("Network error", {
          status: 408,
          statusText: "Network error",
        });
      })
    );
    return;
  }

  evt.respondWith(
    fetch(evt.request, { cache: "no-store" }).catch(() => {
      // Network failed
      return caches.open(CATALOGUE_ASSETS).then((cache) => {
        return cache.match(evt.request).then((response) => {
          // show offline page
          if (!response && evt.request.mode === "navigate") {
            return caches.match("/offline.html");
          }
          return (
            response ||
            new Response("Offline and not cached", {
              status: 404,
              statusText: "Offline and not cached",
            })
          );
        });
      });
    })
  );
});

self.addEventListener("message", (event) => {
  if (event.data && event.data.type === "CLEAR_CACHE") {
    event.waitUntil(
      caches.keys().then((cacheNames) => {
        return Promise.all(
          cacheNames.map((cacheName) => {
            return caches.delete(cacheName);
          })
        );
      })
    );
  }
});
