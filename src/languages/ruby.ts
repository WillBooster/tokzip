import { rubyModule } from '../generated/ruby.ts';
import { registerLanguageModule } from '../moduleRegistry.ts';

// Self-registers on import; tables are validated at registration.
registerLanguageModule(rubyModule);

export { rubyModule };
