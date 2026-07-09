import { cppModule } from '../generated/cpp.ts';
import { registerLanguageModule } from '../moduleRegistry.ts';

// Self-registers on import; tables are validated at registration.
registerLanguageModule(cppModule);

export { cppModule };
