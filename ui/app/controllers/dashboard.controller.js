(function () {
  "use strict";

  angular.module("inPhaseApp").controller("DashboardController", [
    "ApiService",
    "$location",
    "$q",
    "$rootScope",
    function (ApiService, $location, $q, $rootScope) {
      var vm = this;

      vm.loading = true;
      vm.error = null;
      vm.summary = null;
      vm.readiness = null;
      vm.ndtSamples = [];
      vm.busiRows = [];
      vm.isAdmin = ApiService.hasRole("admin");
      vm.opsSummary = null;
      vm.opsRecent = [];
      vm.clientErrors = ApiService.getClientErrors();

      vm.go = function (path) {
        $location.path(path);
      };

      vm.refreshClientErrors = function () {
        vm.clientErrors = ApiService.getClientErrors();
      };

      vm.clearClientErrors = function () {
        ApiService.clearClientErrors();
        vm.clientErrors = [];
      };

      function load() {
        vm.loading = true;
        vm.error = null;

        var requests = [
          ApiService.getDashboardSummary(),
          ApiService.getDashboardReadiness(),
          ApiService.listNdtSamples(),
        ];
        if (vm.isAdmin) {
          requests.push(ApiService.getOpsErrorSummary(24 * 60));
          requests.push(ApiService.getOpsErrorRecent(12));
        }

        $q.all(requests)
          .then(function (responses) {
            vm.summary = responses[0];
            vm.readiness = responses[1];
            vm.ndtSamples = responses[2].slice(0, 5);
            if (vm.isAdmin) {
              vm.opsSummary = responses[3];
              vm.opsRecent = responses[4];
            }

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
            vm.error = error.detail || "Failed to load dashboard summary";
          })
          .finally(function () {
            vm.loading = false;
          });
      }

      vm.refresh = load;

      $rootScope.$on("api:error", function () {
        vm.refreshClientErrors();
      });

      load();
    },
  ]);
})();
