(function () {
  "use strict";

  angular.module("inPhaseApp").component("metricTile", {
    bindings: {
      label: "@",
      value: "<",
      tone: "@",
    },
    template:
      '<article class="metric-tile {{$ctrl.tone || \'default\'}}">' +
      '<h4>{{$ctrl.label}}</h4>' +
      '<p>{{$ctrl.value}}</p>' +
      "</article>",
  });
})();
