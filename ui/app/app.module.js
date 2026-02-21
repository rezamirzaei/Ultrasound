(function () {
  "use strict";

  angular
    .module("inPhaseApp", ["ngRoute"])
    .run([
      "$rootScope",
      "$location",
      "ApiService",
      function ($rootScope, $location, ApiService) {
        $rootScope.$on("$routeChangeStart", function (event, next) {
          var requiredRole = next && next.$$route ? next.$$route.requiredRole : null;
          if (!requiredRole) {
            return;
          }

          if (!ApiService.hasRole(requiredRole)) {
            event.preventDefault();
            $location.path("/dashboard");
          }
        });
      },
    ]);
})();
