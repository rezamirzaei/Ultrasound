(function () {
  "use strict";

  angular.module("inPhaseApp").service("ApiService", [
    "$http",
    "$q",
    function ($http, $q) {
      var baseUrl = "/api/v1";

      function unwrap(promise) {
        return promise.then(
          function (response) {
            return response.data;
          },
          function (error) {
            var detail = (error.data && error.data.detail) || "Request failed";
            return $q.reject({
              status: error.status || 0,
              detail: detail,
            });
          }
        );
      }

      this.getBaseUrl = function () {
        return baseUrl;
      };

      this.health = function () {
        return unwrap($http.get(baseUrl + "/health"));
      };

      this.getDashboardSummary = function () {
        return unwrap($http.get(baseUrl + "/dashboard/summary"));
      };

      this.getDashboardReadiness = function () {
        return unwrap($http.get(baseUrl + "/dashboard/readiness"));
      };

      this.getBusiCounts = function () {
        return unwrap($http.get(baseUrl + "/datasets/busi/counts"));
      };

      this.getBusiSamplePreview = function (className, sampleIndex) {
        return unwrap(
          $http.get(
          baseUrl +
            "/datasets/busi/samples/" +
            encodeURIComponent(className) +
            "/" +
            encodeURIComponent(String(sampleIndex))
          )
        );
      };

      this.listNdtSamples = function () {
        return unwrap($http.get(baseUrl + "/datasets/ndt/samples"));
      };

      this.getNdtSample = function (sampleName) {
        return unwrap($http.get(baseUrl + "/datasets/ndt/samples/" + encodeURIComponent(sampleName)));
      };

      this.getNdtSignal = function (sampleName, maxPoints) {
        return unwrap(
          $http.get(baseUrl + "/datasets/ndt/samples/" + encodeURIComponent(sampleName) + "/signal", {
            params: { max_points: maxPoints || 1024 },
          })
        );
      };

      this.previewPreprocessing = function (payload) {
        return unwrap($http.post(baseUrl + "/preprocessing/preview", payload));
      };
    },
  ]);
})();
