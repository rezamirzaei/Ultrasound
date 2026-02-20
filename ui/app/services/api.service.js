(function () {
  "use strict";

  angular.module("inPhaseApp").service("ApiService", [
    "$http",
    function ($http) {
      var baseUrl = "/api/v1";

      this.getBaseUrl = function () {
        return baseUrl;
      };

      this.health = function () {
        return $http.get(baseUrl + "/health");
      };

      this.getDashboardSummary = function () {
        return $http.get(baseUrl + "/dashboard/summary");
      };

      this.getBusiCounts = function () {
        return $http.get(baseUrl + "/datasets/busi/counts");
      };

      this.listNdtSamples = function () {
        return $http.get(baseUrl + "/datasets/ndt/samples");
      };

      this.getNdtSample = function (sampleName) {
        return $http.get(baseUrl + "/datasets/ndt/samples/" + encodeURIComponent(sampleName));
      };

      this.previewPreprocessing = function (payload) {
        return $http.post(baseUrl + "/preprocessing/preview", payload);
      };
    },
  ]);
})();
