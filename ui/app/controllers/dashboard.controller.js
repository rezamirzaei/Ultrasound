(function () {
  "use strict";

  angular.module("inPhaseApp").controller("DashboardController", [
    "ApiService",
    function (ApiService) {
      var vm = this;

      vm.loading = true;
      vm.error = null;
      vm.summary = null;

      function load() {
        vm.loading = true;
        vm.error = null;

        ApiService.getDashboardSummary()
          .then(function (response) {
            vm.summary = response.data;
          })
          .catch(function (error) {
            vm.error = (error.data && error.data.detail) || "Failed to load dashboard summary";
          })
          .finally(function () {
            vm.loading = false;
          });
      }

      vm.refresh = load;
      load();
    },
  ]);
})();
