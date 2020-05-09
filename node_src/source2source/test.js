// here is the comments for testing
const x = function(grammar, replacementScope) {
    grammar.name = 'JSON with comments';
    grammar.scopeName = `source${replacementScope}`;
 
    var fixScopeNames = function(rule) {
        if (typeof rule.name === 'string') {
            rule.name = rule.name.replace(/\\.json/g, replacementScope);
        }
        if (typeof rule.contentName === 'string') {
            rule.contentName = rule.contentName.replace(/\\.json/g, replacementScope);
        }
        for (var property in rule) {
            var value = rule[property];
            if (typeof value === 'object') {
                fixScopeNames(value);
            }
        }
    };
    
    var repository = grammar.repository;
    for (var key in repository) {
        fixScopeNames(repository[key]);
    }
}
