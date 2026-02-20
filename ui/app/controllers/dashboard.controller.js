(function () {
  "use strict";

  angular.module("inPhaseApp").controller("DashboardController", [
    "ApiService",
    "$location",
    "$q",
    function (ApiService, $location, $q) {
      var vm = this;

      vm.loading = true;
      vm.error = null;
      vm.summary = null;
      vm.ndtSamples = [];
      vm.busiRows = [];

      vm.go = function (path) {
        $location.path(path);
      };

      function load() {
        vm.loading = true;
        vm.error = null;

        $q.all([ApiService.getDashboardSummary(), ApiService.listNdtSamples()])
          .then(function (responses) {
            vm.summary = responses[0].data;
            vm.ndtSamples = responses[1].data.slice(0, 5);
            vm.busiRows = Object.keys(vm.summary.busi_counts || {})
              .map(function (className) {
                var count = vm.summary.busi_counts[className] || 0;
                var pct = vm.summary.busi_total > 0 ? (count / vm.summary.busi_total) * 100 : 0;
                return {
                  name: className,
                  count: count,
                  pct: pct,
                };
              })
              .sort(function (left, right) {
                return right.count - left.count;
              });
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
