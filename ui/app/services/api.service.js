(function () {
  "use strict";

  angular.module("inPhaseApp").service("ApiService", [
    "$http",
    "$q",
    "$rootScope",
    function ($http, $q, $rootScope) {
      var baseUrl = "/api/v1";
      var sessionStorageKey = "inphase.auth.session";
      var roleRank = { viewer: 1, analyst: 2, admin: 3 };
      var authSession = null;
      var clientErrors = [];

      function loadSession() {
        if (!window.localStorage) {
          return null;
        }
        try {
          var raw = window.localStorage.getItem(sessionStorageKey);
          if (!raw) {
            return null;
          }
          var parsed = JSON.parse(raw);
          if (!parsed || !parsed.access_token || !parsed.role || !parsed.username) {
            return null;
          }
          return parsed;
        } catch (error) {
          return null;
        }
      }

      function saveSession(session) {
        authSession = session;
        if (!window.localStorage) {
          return;
        }
        try {
          window.localStorage.setItem(sessionStorageKey, JSON.stringify(session));
        } catch (error) {
          // Ignore storage failures in private browsing contexts.
        }
      }

      function clearSession() {
        authSession = null;
        if (!window.localStorage) {
          return;
        }
        try {
          window.localStorage.removeItem(sessionStorageKey);
        } catch (error) {
          // Ignore storage failures in private browsing contexts.
        }
      }

      function withAuth(config) {
        var merged = angular.extend({}, config || {});
        merged.headers = angular.extend({}, merged.headers || {});
        if (authSession && authSession.access_token) {
          merged.headers.Authorization = "Bearer " + authSession.access_token;
        }
        return merged;
      }

      function recordClientError(error) {
        var statusCode = error.status || 0;
        var detail = (error.data && error.data.detail) || "Request failed";
        var requestId = null;
        try {
          if (typeof error.headers === "function") {
            requestId = error.headers("x-request-id") || null;
          }
        } catch (headerError) {
          requestId = null;
        }
        var path = (error.config && error.config.url) || "";
        var event = {
          occurred_at: new Date().toISOString(),
          request_id: requestId,
          method: (error.config && error.config.method && error.config.method.toUpperCase()) || "UNKNOWN",
          path: path,
          status_code: statusCode,
          detail: detail,
          role: authSession ? authSession.role : null,
        };

        clientErrors.unshift(event);
        if (clientErrors.length > 100) {
          clientErrors.length = 100;
        }
        $rootScope.$broadcast("api:error", event);
      }

      function unwrap(promise) {
        return promise.then(
          function (response) {
            return response.data;
          },
          function (error) {
            recordClientError(error);

            var detail = (error.data && error.data.detail) || "Request failed";
            var requestId = (error.data && error.data.request_id) || null;
            var statusCode = error.status || 0;

            if ((statusCode === 401 || statusCode === 403) && authSession) {
              clearSession();
              $rootScope.$broadcast("auth:expired");
            }

            return $q.reject({
              status: statusCode,
              detail: detail,
              request_id: requestId,
            });
          }
        );
      }

      authSession = loadSession();

      this.getBaseUrl = function () {
        return baseUrl;
      };

      this.isAuthenticated = function () {
        return !!(authSession && authSession.access_token);
      };

      this.getAuthSession = function () {
        return authSession ? angular.copy(authSession) : null;
      };

      this.hasRole = function (requiredRole) {
        if (!authSession || !authSession.role) {
          return false;
        }
        return (roleRank[authSession.role] || 0) >= (roleRank[requiredRole] || 99);
      };

      this.login = function (username, password) {
        return unwrap($http.post(baseUrl + "/auth/login", { username: username, password: password })).then(
          function (session) {
            saveSession(session);
            return session;
          }
        );
      };

      this.logout = function () {
        if (!authSession || !authSession.access_token) {
          clearSession();
          return $q.when({ success: true, revoked_token: false });
        }

        return $http
          .post(baseUrl + "/auth/logout", {}, withAuth())
          .then(
            function (response) {
              return response.data;
            },
            function () {
              return { success: true, revoked_token: false };
            }
          )
          .finally(function () {
            clearSession();
          });
      };

      this.me = function () {
        return unwrap($http.get(baseUrl + "/auth/me", withAuth())).then(function (profile) {
          if (authSession) {
            authSession.username = profile.username;
            authSession.role = profile.role;
            authSession.expires_at = profile.expires_at;
            saveSession(authSession);
          }
          return profile;
        });
      };

      this.getClientErrors = function () {
        return clientErrors.slice(0, 50);
      };

      this.clearClientErrors = function () {
        clientErrors = [];
      };

      this.health = function () {
        return unwrap($http.get(baseUrl + "/health"));
      };

      this.getDashboardSummary = function () {
        return unwrap($http.get(baseUrl + "/dashboard/summary", withAuth()));
      };

      this.getDashboardReadiness = function () {
        return unwrap($http.get(baseUrl + "/dashboard/readiness", withAuth()));
      };

      this.getOpsErrorSummary = function (windowMinutes) {
        return unwrap(
          $http.get(baseUrl + "/ops/errors/summary", withAuth({ params: { window_minutes: windowMinutes || 60 } }))
        );
      };

      this.getOpsErrorRecent = function (limit) {
        return unwrap(
          $http.get(baseUrl + "/ops/errors/recent", withAuth({ params: { limit: limit || 20 } }))
        );
      };

      this.resyncDatasets = function () {
        return unwrap($http.post(baseUrl + "/ops/datasets/resync", {}, withAuth()));
      };

      this.getBusiCounts = function () {
        return unwrap($http.get(baseUrl + "/datasets/busi/counts", withAuth()));
      };

      this.getIndustrialSummary = function () {
        return unwrap($http.get(baseUrl + "/datasets/industrial/summary", withAuth()));
      };

      this.getIndustrialSamplePreview = function (datasetName, split, className, sampleIndex) {
        return unwrap(
          $http.get(
            baseUrl +
              "/datasets/industrial/samples/" +
              encodeURIComponent(datasetName) +
              "/" +
              encodeURIComponent(split) +
              "/" +
              encodeURIComponent(className) +
              "/" +
              encodeURIComponent(String(sampleIndex)),
            withAuth()
          )
        );
      };

      this.getIndustrialSegmentationPreview = function (datasetName, split, className, sampleIndex) {
        return unwrap(
          $http.get(
            baseUrl +
              "/datasets/industrial/segmentation/" +
              encodeURIComponent(datasetName) +
              "/" +
              encodeURIComponent(split) +
              "/" +
              encodeURIComponent(className) +
              "/" +
              encodeURIComponent(String(sampleIndex)),
            withAuth()
          )
        );
      };

      this.getIndustrialTrainingLatest = function (datasetName) {
        return unwrap(
          $http.get(
            baseUrl + "/datasets/industrial/training/latest",
            withAuth({ params: { dataset_name: datasetName || "steel_defect" } })
          )
        );
      };

      this.runIndustrialTraining = function (payload) {
        return unwrap($http.post(baseUrl + "/datasets/industrial/training/run", payload, withAuth()));
      };

      this.getBusiSamplePreview = function (className, sampleIndex) {
        return unwrap(
          $http.get(
            baseUrl + "/datasets/busi/samples/" + encodeURIComponent(className) + "/" + encodeURIComponent(String(sampleIndex)),
            withAuth()
          )
        );
      };

      this.getBusiTrainingLatest = function (includeNormal) {
        return unwrap(
          $http.get(
            baseUrl + "/datasets/busi/training/latest",
            withAuth({ params: { include_normal: !!includeNormal } })
          )
        );
      };

      this.runBusiTraining = function (payload) {
        return unwrap($http.post(baseUrl + "/datasets/busi/training/run", payload, withAuth()));
      };

      this.enqueueBusiTrainingJob = function (payload) {
        return unwrap($http.post(baseUrl + "/learning/jobs/busi-training", payload, withAuth()));
      };

      this.enqueueDatasetResyncJob = function () {
        return unwrap($http.post(baseUrl + "/learning/jobs/datasets-resync", {}, withAuth()));
      };

      this.enqueueIndustrialTrainingJob = function (payload) {
        return unwrap($http.post(baseUrl + "/learning/jobs/industrial-training", payload, withAuth()));
      };

      this.listLearningJobs = function (limit) {
        return unwrap($http.get(baseUrl + "/learning/jobs", withAuth({ params: { limit: limit || 30 } })));
      };

      this.getLearningJob = function (jobId) {
        return unwrap($http.get(baseUrl + "/learning/jobs/" + encodeURIComponent(String(jobId)), withAuth()));
      };

      this.getDatabaseSchemaStatus = function () {
        return unwrap($http.get(baseUrl + "/ops/database/schema-status", withAuth()));
      };

      this.uploadBusiSample = function (formData) {
        return unwrap(
          $http.post(
            baseUrl + "/datasets/busi/upload",
            formData,
            withAuth({
              transformRequest: angular.identity,
              headers: { "Content-Type": undefined },
            })
          )
        );
      };

      this.uploadIndustrialSample = function (formData) {
        return unwrap(
          $http.post(
            baseUrl + "/datasets/industrial/upload",
            formData,
            withAuth({
              transformRequest: angular.identity,
              headers: { "Content-Type": undefined },
            })
          )
        );
      };

      this.listNdtSamples = function () {
        return unwrap($http.get(baseUrl + "/datasets/ndt/samples", withAuth()));
      };

      this.getNdtSample = function (sampleName) {
        return unwrap($http.get(baseUrl + "/datasets/ndt/samples/" + encodeURIComponent(sampleName), withAuth()));
      };

      this.getNdtSignal = function (sampleName, maxPoints) {
        return unwrap(
          $http.get(
            baseUrl + "/datasets/ndt/samples/" + encodeURIComponent(sampleName) + "/signal",
            withAuth({ params: { max_points: maxPoints || 1024 } })
          )
        );
      };

      this.previewPreprocessing = function (payload) {
        return unwrap($http.post(baseUrl + "/preprocessing/preview", payload, withAuth()));
      };
    },
  ]);
})();
